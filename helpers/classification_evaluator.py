import os
import multiprocessing
import torch

from torch.utils.data import DataLoader

from torchvision import transforms

from utils import Accuracy
from datasets import Customdataset

class ClassificationEvaluator:
    def __init__(self, args, model):
        self.args = args

        self.model = model

        self.dataset = self.init_dataset()
        self.dataloader = self.init_dataloader()

        self.data_cnt = len(self.dataset)
        self.remainder = self.data_cnt%args.batch_size
        self.metric = Accuracy(topk=args.topk)
        self.topk = args.topk



    def __call__(self):
        args = self.args

        self.model.eval()
        self.metric.reset()
        self.class_correct = list(0. for i in range(len(self.class_names)))
        self.class_total = list(0. for i in range(len(self.class_names)))
            
        cnt = 0

        for i, batch in enumerate(self.dataloader):
            x, y = batch

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

            with torch.no_grad():
                outputs = self.model(x)
                pred = torch.exp(outputs)
                top_prob,top_class = pred.topk(self.topk, 1)
                y_ = y.unsqueeze(1).expand_as(top_class)
                c = (top_class == y_).squeeze()
                if(len(y) < args.batch_size):
                    for i in range(self.remainder):
                        label = y[i]
                        self.class_correct[label] += c[i].sum().item()
                        self.class_total[label] += 1 
                else:
                    for i in range(args.batch_size):
                        label = y[i]
                        self.class_correct[label] += c[i].sum().item()
                        self.class_total[label] += 1
                self.match(self.model(x), y)

            cnt += x.size(0)

        print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

        accuracy = self.metric.get_result()

        # print results
        print("accuracy = %.3f %%" % accuracy)
        for i in range(len(self.class_names)):
            print('Accuracy of %s: %.3f %%' % (self.class_names[i], 100 * self.class_correct[i] / self.class_total[i]))

    def match(self, y_, y):
        self.metric.match(y_, y)

    def init_dataset(self):
        args = self.args

        t = []

        t.extend([transforms.Resize((args.input_size[1],args.input_size[0])),
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

        return DataLoader(self.dataset,
                          pin_memory=True,
                          batch_size=args.batch_size,
                          num_workers=num_workers)


