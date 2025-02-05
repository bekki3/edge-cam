import os
import multiprocessing
import torch

from torch import nn
from torch.optim import SGD,Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau

from torch.utils.data import DataLoader

from torchvision import transforms
from datasets import Customdataset

from models import build_model, load_model, save_model
from datasets import load_dataset

class ClassificationTrainer:
    def __init__(self, args, callback=None):
        self.args = args
        self.dataset = self.init_dataset()
        self.dataloader = self.init_dataloader()
        self.model = self.init_model()
        self.criterion = self.init_criterion()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.epoch = 0
        self.loss = 0
        if args.resume:
            self.epoch,self.loss=self.load_model(args.resume)
            print("\nresume at epoch: {}, loss: {}\n".format(self.epoch,self.loss))
            self.epoch = self.epoch + 1
        self.callback = callback

    def fit(self):
        if self.callback:
            self.callback.fit_start(self)

        model = self.model

        model.train()

        for epoch in range(self.epoch, self.args.epochs):
            loss = self.step(model, epoch)

        if self.callback:
            size = self.size
            self.callback.fit_end(self, loss, size,epoch)

    def step(self, model, epoch):
        if self.callback:
            self.callback.step_start(self, epoch)

        losses = []

        for i, batch in enumerate(self.dataloader):
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
            x = x.cuda()
            y = y.cuda()

        y_ = model.forward(x)
        loss = self.criterion(y_, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss = loss.item()

        if self.callback:
            self.callback.minibatch_end(self, epoch, idx, loss)

        return loss

    def init_dataset(self):
        args = self.args

        self.size = args.input_size
        print(self.size)

        t = [transforms.Resize((self.size[1],self.size[0]))]

        t.extend([transforms.RandomHorizontalFlip(p=0.2),
                  transforms.RandomGrayscale(p=0.1),
                  transforms.RandomPerspective(distortion_scale=0.2),
                  transforms.RandomVerticalFlip(p=0.2),
                  transforms.ToTensor(),
                  transforms.Normalize(args.mean, args.std)])

        t = transforms.Compose(t)

        dataset = Customdataset.CustomDataset(data_set_path=args.dataset_root,transforms=t)
        self.class_names = os.walk(args.dataset_root).__next__()[1]
        self.class_names.sort()
        return dataset

    def init_dataloader(self):
        args = self.args

        if args.num_workers < 0:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = args.num_workers

        dataloader = DataLoader(self.dataset,
                                pin_memory=True,
                                shuffle=True,
                                drop_last=True,
                                batch_size=args.batch_size,
                                num_workers=num_workers)

        return dataloader
    
    def init_model(self):
        args = self.args

        pretrained = True if not args.resume else False

        model = build_model(args,custom_class_num=len(self.class_names), pretrained=pretrained)

        return model

    def init_criterion(self):
        loss = nn.CrossEntropyLoss()

        return loss

    def init_optimizer(self):
        args = self.args

        if args.optimizer == "sgd":
            optimizer = SGD(self.model.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
        elif args.optimizer == "adam":
            optimizer = Adam(self.model.parameters())
        else:
            raise Exception("unknown optimizer")
        return optimizer

    def init_scheduler(self):
        if self.args.scheduler == "step":
            return StepLR(self.optimizer,
                          self.args.step_size,
                          self.args.gamma)

        elif self.args.scheduler == "multi_step":
            return MultiStepLR(self.optimizer,
                               self.args.milestones,
                               self.args.gamma)

        elif self.args.scheduler == "plateau":
            return ReduceLROnPlateau(self.optimizer,mode='min',patience=5)
            
        else:
            return None

    def load_model(self, filename):
        epoch, loss= load_model(self.model, filename,self.optimizer)
        return epoch, loss


    def save_model(self, epoch, loss, optimizer, path='./checkpoints', postfix=None):
        save_model(self.model,epoch,loss,optimizer, path, postfix)

