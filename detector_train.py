import argparse
import os
from helpers import DetectionTrainer
from utils import TrainerCallback
from datetime import datetime
from tensorboardX import SummaryWriter 

class Callback(TrainerCallback):
    def __init__(self, args):
        super().__init__()

        # Print training configuration
        print("Start Training")
        print("Model: %s" % args.model)
        print(f"Epochs : {args.epochs}")
        print("Batch Size: %d" % args.batch_size)
        print("Learning Rate: %.5f" % args.lr)
        if args.optimizer == "sgd":
            print("Optimizer : SGD")
            print("Momentum: %.1f" % args.momentum)
        elif args.optimizer == "adam":
            print("Optimizer : Adam")
        elif args.optimizer == "adamw":
            print("Optimizer : AdamW")
        else:
            print("Optimizer: None")
        
        if args.scheduler == "step":
            print("Scheduler: Step LR scheduler")
            print("Step Size: %d" % args.step_size)
            print("Gamma: %.5f" % args.gamma)
        elif args.scheduler == "multi_step":
            milestones = [str(m) for m in args.milestones]
            print("Scheduler: Multi-step LR scheduler")
            print("Milestones: %s" % ", ".join(milestones))
            print("Gamma: %.5f" % args.gamma)
        elif args.scheduler == "plateau":
            print("Scheduler: Plateau LR scheduler")
        elif args.scheduler == "cosine":
            print("Scheduler: Cosine Annealing LR scheduler")
        else:
            print("Scheduler: None")

        print("")

        self.min_loss = 99.99
        self.batch_size = args.batch_size
        self.total_batches = 0

        # Initialize tensorboard writer
        self.summary = SummaryWriter()

    def fit_start(self, t):
        self.total_cnt = len(t.dataset)

    def step_start(self, t, epoch):
        print(str(datetime.now()) + "  epoch#%d start" % epoch)

    def minibatch_end(self, trainer, epoch, idx, loss):
        cnt = (idx + 1) * self.batch_size
        print("[%d / %d] - loss=%.3f" % (cnt, self.total_cnt, loss), end='\r')

    def step_end(self, t, epoch, loss, args):
        print(str(datetime.now()) + "  epoch#%d end - mean loss=%.3f" % (epoch, loss))
        self.min_loss = min(self.min_loss, loss)
        self.summary.add_scalar('loss', loss, epoch)
        self.summary.add_scalar('min_loss', self.min_loss, epoch)
        for param_group in t.optimizer.param_groups:
            self.summary.add_scalar('lr', param_group['lr'], epoch)
        # Save checkpoint models
        if self.min_loss == loss:
            t.save_model(epoch=epoch, loss=self.min_loss, optimizer=t.optimizer, postfix='obj_best')
        elif epoch % args.checkpoint_every == 0:
            t.save_model(epoch=epoch, loss=loss, optimizer=t.optimizer, postfix='obj_check')

    def fit_end(self, t, loss, epoch):
        t.save_model(epoch=epoch, loss=loss, optimizer=t.optimizer, postfix='obj_latest')
        self.summary.close()


def main():
    parser = argparse.ArgumentParser(description='Detector Training')
    parser.add_argument('--dataset_root', default='./PASCAL_VOC_Dataset/PASCAL_VOC_2007/,./PASCAL_VOC_Dataset/PASCAL_VOC_2012/',
                        help='Dataset root directory path') #'./PASCAL_VOC_Dataset/PASCAL_VOC_2007/,./PASCAL_VOC_Dataset/PASCAL_VOC_2012/'
    parser.add_argument('--dataset_domains', default='train,validation', type=str,
                        help='Dataset domains (comma separated)')
    parser.add_argument('--class_path', default='./PASCAL_VOC.txt', type=str,
                        help='Class label txt file directory')
    parser.add_argument('--model', default='ssdlite640', choices=['ssdlite320','ssdlite512','ssdlite416','ssdlite640'],
                        help='Detector model name')
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training (Enter a value greater than 3)') #32 6.9GB
    parser.add_argument('--checkpoint_every', default=10, type=int, 
                        help='Save a checkpoint every N epochs during training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers', default=6, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--no-preload', dest='preload', action='store_false',
                        help='Do not preload dataset into memory')
    parser.add_argument('--prefetch_factor', default=2, type=int,
                        help='Factor of number of workers used in dataloading')
    parser.add_argument('--epochs', default=600, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--th_iou', default=0.5, type=float,
                        help='IOU Threshold')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        help='Initial learning rate') #1e-3
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for SGD')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--step_size', default=150, type=int,
                        help='Step size for step LR scheduler')
    parser.add_argument('--milestones', default=[80, 140, 170], type=int, nargs='*',
                        help='Milestones for multi-step LR scheduler')
    parser.add_argument('--disable_augmentation', default=False, 
                        action='store_true',
                        help='Disable random augmentation')
    parser.add_argument('--enable_letterbox', default=False, 
                        action='store_true',
                        help='Enable letterboxing image')
    parser.add_argument('--scheduler', default='cosine',
                        choices=['plateau', 'step', 'multi_step', 'cosine'],
                        type=str.lower, help='Learning rate scheduler')
    parser.add_argument('--optimizer', default='adamw',
                        choices=['adam', 'sgd', 'adamw'],
                        type=str.lower, help='Optimizer choice')
    parser.add_argument('--disable_random_expand', default=False, 
                        action='store_true',
                        help='Disable random expansion augmentation')
    parser.add_argument('--feature', default=False, 
                        action='store_true',
                        help='Extract feature maps for ssdlite512 (64*64 size)')
    args = parser.parse_args()

    # Check class_path exists
    if not os.path.isfile(args.class_path):
        raise Exception("There is no label data")
    # Check batch_size validity
    if args.batch_size < 4:
        raise Exception("Please enter a value greater than 3 for BATCH_SIZE")

    trainer = DetectionTrainer(args, callback=Callback(args))
    trainer.fit()


if __name__ == "__main__":
    main()
