import argparse

from helpers import ClassificationTrainer
from utils import TrainerCallback
import os
from tensorboardX import SummaryWriter


class Callback(TrainerCallback):
    def __init__(self, args):
        super().__init__()

        # print status
        print("Start Training")
        print("Model: %s" % args.model)
        print("Batch Size: %d" % args.batch_size)
        print("Learning Rate: %.3f" % args.lr)
        if args.optimizer == "sgd":
            print("Optimizer : SGD")
            print("Momentum: %.1f" % args.momentum)
        elif args.optimizer == "adam":
            print("Optimizer : Adam")
        else:
        	print("Optimizer: None")

        if args.scheduler == "step":
            print("Scheduler: Step LR scheduler")
            print("Step Size: %d" % args.step_size)
            print("Gamma: %.2f" % args.gamma)
        elif args.scheduler == "multi_step":
            milestones = [str(m) for m in args.milestones]
            print("Scheduler: Multi-step LR scheduler")
            print("Milestones: %s" % ", ".join(milestones))
            print("Gamma: %.2f" % args.gamma)
        elif args.scheduler == "plateau":
            print("Scheduler: Plateau LR scheduler")
        else:
            print("Scheduler: None")

        print("")

        self.min_loss = 99.99

        self.batch_size = args.batch_size
        self.total_batches = 0
        # init tensorboard
        self.summary = SummaryWriter()

    def fit_start(self, t):
        batch_size = self.batch_size
        self.total_cnt = len(t.dataset)

    def step_start(self, t, epoch):
        print("epoch#%d start" % epoch)

    def minibatch_end(self, trainer, epoch, idx, loss):
        cnt = (idx + 1) * self.batch_size
        total_cnt = self.total_cnt

        print("[%d / %d] - loss=%.3f" % (cnt, total_cnt, loss), end='\r')

    def step_end(self, t, epoch, loss):
        print("epoch#%d end - mean loss=%.3f" % (epoch, loss))

        # write log
        self.min_loss = min(self.min_loss, loss)

        self.summary.add_scalar('loss', loss, epoch)
        self.summary.add_scalar('min_loss', self.min_loss, epoch)

        for param_group in t.optimizer.param_groups:
            self.summary.add_scalar('lr', param_group['lr'], epoch)

        # save model
        if(epoch%5 == 0):
            t.save_model(epoch=epoch,loss=loss,optimizer=t.optimizer,postfix='class_check')

        if(self.min_loss == loss):
            t.save_model(epoch=epoch,loss=self.min_loss,optimizer=t.optimizer,postfix='class_best')

    def fit_end(self, t, loss, size, epoch):
        t.save_model(epoch=epoch, loss=loss, optimizer=t.optimizer,postfix='class_latest')
        self.summary.close()


def main():
    parser = argparse.ArgumentParser(description='Classification Training')

    parser.add_argument('--dataset_root', default="./dataset",
                        help='Dataset root directory path')
    parser.add_argument('--model', default='mobilenetv2',choices=['mobilenetv2'],
                        help='Detector model name')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optimizer')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--step_size', default=70, type=int,
                        help='Step size for step lr scheduler')
    parser.add_argument('--milestones', default=[110, 150], type=int, nargs='*',
                        help='Milestones for multi step lr scheduler')
    parser.add_argument('--disable_augmentation', default=False, 
                        action='store_true',
                        help='Disable random augmentation')
    parser.add_argument('--scheduler', default='multi_step',
                        choices=['Plateau','step', 'multi_step'],
                        type=str, help='Use Scheduler')
    parser.add_argument('--optimizer', default='sgd',
                        choices=['adam', 'sgd'],
                        type=str.lower, help='Use Optimizer')
    parser.add_argument('--input_size', default=[320,320], type=int,nargs=2,
                        help='input size (width, height)')

    # model parameter
    parser.add_argument('--mean', nargs=3, type=float,
                        default=(0.486, 0.456, 0.406),
                        help='mean for normalizing')
    parser.add_argument('--std', nargs=3, type=float,
                        default=(0.229, 0.224, 0.225),
                        help='std for normalizing')

    args = parser.parse_args()

    # dataset_root check
    if not os.path.isdir(args.dataset_root):
        raise Exception("There is no dataset_root dir") 

    t = ClassificationTrainer(args, callback=Callback(args))
    t.fit()


if __name__ == "__main__":
    main()

