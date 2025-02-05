import os
import multiprocessing
import torch

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import load_dataset
from transforms import detector_transforms as transforms

from models import build_model, load_model, save_model
from losses import build_loss
from utils.txt_utils import read_txt

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
            if batch[0].size(0) < self.args.batch_size:
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
            self.callback.step_end(self, epoch, loss)
        return loss
        
    def minibatch(self, model, epoch, idx, batch):
        if self.callback:
            self.callback.minibatch_start(self, epoch, idx)
        x, y = batch
        if torch.cuda.is_available():
            # Use non_blocking transfer to reduce CPU-GPU sync overhead
            x = x.to(self.device, non_blocking=True)
        y_pred = model.forward(x)
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_val = loss.item()
        if self.callback:
            self.callback.minibatch_end(self, epoch, idx, loss_val)
        return loss_val

    def init_dataset(self):
        args = self.args
        size = self.model.get_input_size()
        t = []
        if not args.disable_augmentation:
            t.extend([transforms.RandomSamplePatch()])
        if not args.disable_random_expand:
            t.extend([transforms.RandomExpand()])   
        if args.enable_letterbox:
            t.append(transforms.LetterBox())
        t.append(transforms.Resize(size))
        if not args.disable_augmentation:
            t.extend([transforms.RandomHorizontalFlip(),
                      transforms.RandomDistort(),
                      transforms.RandomColorSpace()])
        t.extend([transforms.ToTensor(),
                   transforms.Normalize()])
        t = transforms.Compose(t)
        dataset = load_dataset(args, transforms=t, eval_mode_value=self.eval_mode_value)
        return dataset

    def init_dataloader(self):
        args = self.args
        num_workers = multiprocessing.cpu_count() if args.num_workers < 0 else args.num_workers
        # Add persistent_workers to keep worker processes alive across epochs.
        dataloader = DataLoader(self.dataset,
                                pin_memory=True,
                                shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=num_workers,
                                collate_fn=self.collate,
                                drop_last=True,
                                prefetch_factor=8,
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
        imgs = []
        targets = []
        for (img, target) in batch:
            imgs.append(img)
            targets.append(target)
        return torch.stack(imgs, 0), targets
