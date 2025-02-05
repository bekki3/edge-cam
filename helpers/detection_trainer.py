import os
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import load_dataset
from transforms import detector_transforms as transforms

from models import build_model, load_model, save_model
from losses import build_loss
from utils.txt_utils import read_txt

# Placeholder GPU transforms (replace with your GPU implementations if available)
class RandomExpandGPU(nn.Module):
    def __init__(self, probability=0.5):
        super(RandomExpandGPU, self).__init__()
        self.probability = probability

    def forward(self, x):
        # Placeholder: Implement GPU-based random expansion.
        # For now, simply return x unchanged.
        return x

class LetterBoxGPU(nn.Module):
    def __init__(self, target_size):
        super(LetterBoxGPU, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        # Placeholder: Implement GPU-based letterbox transform.
        # For now, simply perform a resize as a placeholder.
        return K.Resize(self.target_size)(x)

class DetectionTrainer:
    def __init__(self, args, callback=None):
        self.args = args
        self.eval_mode_value = 0
        class_names = read_txt(args.class_path)
        self.num_classes = len(class_names)
        if self.num_classes == 3:
            raise Exception("The number of classes should not be three.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()
        # Initialize GPU-based augmentation pipeline after getting input size from model
        input_size = self.model.get_input_size()
        self.gpu_transforms = self.init_gpu_transforms(input_size)
        # Initialize dataset with minimal CPU transforms (only ToTensor and Normalize)
        self.dataset = self.init_dataset()
        self.dataloader = self.init_dataloader()
        self.criterion = self.init_criterion()
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.epoch = 0
        self.loss = 0
        if args.resume:
            self.epoch, self.loss = self.load_model(args.resume)
            print("\nResuming at epoch: {}, loss: {}\n".format(self.epoch, self.loss))
            self.epoch += 1
        self.callback = callback

    def init_gpu_transforms(self, input_size):
        transforms_list = []
        # Augmentations controlled by disable_augmentation
        if not self.args.disable_augmentation:
            transforms_list.append(K.RandomHorizontalFlip(p=0.5))
            transforms_list.append(K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
            # You can also add additional GPU-friendly augmentations here, such as distortions if available.
        
        # Random expand controlled by disable_random_expand
        if not self.args.disable_random_expand:
            transforms_list.append(RandomExpandGPU(probability=0.5))
        
        # Letterbox transformation controlled by enable_letterbox
        if self.args.enable_letterbox:
            transforms_list.append(LetterBoxGPU(input_size))
        
        # Always ensure final resizing to the model's expected input size.
        transforms_list.append(K.Resize(input_size))
        
        return nn.Sequential(*transforms_list)

    def fit(self):
        if self.callback:
            self.callback.fit_start(self)

        if torch.cuda.is_available():
            model = torch.nn.DataParallel(self.model)
        else:
            model = self.model

        model.train()
        for epoch in range(self.epoch, self.args.epochs):
            loss = self.step(model, epoch)
        if self.callback:
            self.callback.fit_end(self, loss, epoch)

    def step(self, model, epoch):
        if self.callback:
            self.callback.step_start(self, epoch)

        losses = []
        for i, batch in enumerate(self.dataloader):
            # Skip if the batch size is smaller than expected
            # (batch[0] is a list of images)
            if len(batch[0]) < self.args.batch_size:
                continue
            loss = self.minibatch(model, epoch, i, batch)
            losses.append(loss)
        loss = sum(losses) / len(losses)
        if self.scheduler:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(loss)
            else:
                self.scheduler.step()
        if self.callback:
            self.callback.step_end(self, epoch, loss, self.args)
        return loss

    def minibatch(self, model, epoch, idx, batch):
        if self.callback:
            self.callback.minibatch_start(self, epoch, idx)
        imgs, targets = batch
        # Pad images to the same size
        imgs = self.pad_images(imgs)
        # Move to GPU and apply GPU transforms
        imgs = imgs.to(self.device, non_blocking=True)
        imgs = self.gpu_transforms(imgs)
        y_pred = model.forward(imgs)
        loss = self.criterion(y_pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_val = loss.item()
        if self.callback:
            self.callback.minibatch_end(self, epoch, idx, loss_val)
        return loss_val

    @staticmethod
    def pad_images(imgs):
        # Determine maximum height and width among images
        max_h = max(img.shape[1] for img in imgs)
        max_w = max(img.shape[2] for img in imgs)
        padded_imgs = []
        for img in imgs:
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            # Pad (left, right, top, bottom): (0, pad_w, 0, pad_h)
            padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            padded_imgs.append(padded)
        return torch.stack(padded_imgs, dim=0)

    def init_dataset(self):
        args = self.args
        # Minimal CPU transforms: only convert to tensor and normalize.
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize()
        ])
        dataset = load_dataset(args, transforms=t, eval_mode_value=self.eval_mode_value)
        return dataset

    def init_dataloader(self):
        args = self.args
        num_workers = multiprocessing.cpu_count() if args.num_workers < 0 else args.num_workers
        # Use a custom collate function that returns lists instead of stacking.
        dataloader = DataLoader(self.dataset,
                                pin_memory=True,
                                shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=num_workers,
                                collate_fn=self.collate,
                                drop_last=True,
                                prefetch_factor=args.prefetch_factor,
                                persistent_workers=True)
        return dataloader

    def init_model(self):
        args = self.args
        pretrained = True if not args.resume else False
        model = build_model(args, num=self.num_classes, pretrained=pretrained)
        return model

    def init_criterion(self):
        anchor = self.model.get_anchor_box()
        return build_loss(self.args, anchor=anchor)

    def init_optimizer(self):
        args = self.args
        if args.optimizer == "sgd":
            optimizer = SGD(self.model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
        elif args.optimizer == "adam":
            optimizer = Adam(self.model.parameters())
        elif args.optimizer == "adamw":
            optimizer = AdamW(self.model.parameters())
        else:
            raise Exception("Unknown optimizer")
        return optimizer

    def init_scheduler(self):
        if self.args.scheduler == "step":
            return StepLR(self.optimizer, self.args.step_size, self.args.gamma)
        elif self.args.scheduler == "multi_step":
            return MultiStepLR(self.optimizer, self.args.milestones, self.args.gamma)
        elif self.args.scheduler == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode='min', patience=2)
        elif self.args.scheduler == 'cosine':
            return CosineAnnealingLR(self.optimizer, 200, last_epoch=-1)
        else:
            return None

    def load_model(self, filename):
        epoch, loss = load_model(self.model, filename, self.optimizer)
        return epoch, loss

    def save_model(self, epoch, loss, optimizer, path='./checkpoints', postfix=None):
        save_model(self.model, epoch, loss, optimizer, path, postfix)

    @staticmethod
    def collate(batch):
        # Instead of stacking, return lists of images and targets.
        imgs = []
        targets = []
        for (img, target) in batch:
            imgs.append(img)
            targets.append(target)
        return imgs, targets
