import os
import multiprocessing
import torch

from torch.utils.data import DataLoader

from nn import DetectPostProcess
from utils import MeanAp

from datasets import load_dataset
from transforms import detector_transforms as transforms
from utils.txt_utils import read_txt
import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class DetectionEvaluator:
    def __init__(self, args, model):
        self.args = args
        self.eval_mode_value = 1
        self.model = model
        self.post_process = DetectPostProcess(model.get_anchor_box(),
                                              args.th_conf,
                                              args.th_iou)

        self.dataset = self.init_dataset()
        self.dataloader = self.init_dataloader()

        self.data_cnt = len(self.dataset)
        self.label_file_name = read_txt(args.class_path)
        self.num_classes = len(self.label_file_name)
        self.mAP = MeanAp(len(self.label_file_name))

        self.class_names = tuple(self.label_file_name)
        if args.onnx:
            print("use onnx file")
            self.ort_session = onnxruntime.InferenceSession(args.onnx,  providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
            
    def __call__(self):
        args = self.args

        self.model.eval()
        self.mAP.reset()

        cnt = 0

        for i, batch in enumerate(self.dataloader): 
            x, y = batch #image, label

            if torch.cuda.is_available():
                x = x.cuda()

            print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

            with torch.no_grad():
                if args.onnx:
                    ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(x)}
                    ort_outs = self.ort_session.run(None, ort_inputs)
                    ort_outs_conf = torch.as_tensor(ort_outs[0])
                    ort_outs_cls = torch.as_tensor(ort_outs[1])
                    if torch.cuda.is_available():
                        ort_outs_conf = ort_outs_conf.cuda()
                        ort_outs_cls = ort_outs_cls.cuda()
                    y_ = self.post_process((ort_outs_conf, ort_outs_cls))
                else:
                    y_ = self.post_process(self.model(x))

            self.match(y_, y)

            cnt += x.size(0)

        print("[%d / %d]" % (cnt, self.data_cnt), end='\r')

        mAP, aps = self.mAP.calc_mean_ap(self.label_file_name,args.PR_curve)

        # print results
        for cls, ap in enumerate(aps):
            cls_name = self.class_names[cls]

            print("AP(%s) = %.3f" % (cls_name, ap))

        print("mAP = %.3f" % mAP)

    def match(self, y_, y):
        for a, b in zip(y_, y):
            self.mAP.match(a, b)

    def init_dataset(self):
        args = self.args

        size = self.model.get_input_size()

        t = []
        if args.enable_letterbox:
            t.append(transforms.LetterBox())

        t.extend([transforms.Resize(size),
                  transforms.ToTensor(),
                  transforms.Normalize()])

        t = transforms.Compose(t)

        dataset = load_dataset(args,transforms=t, eval_mode_value=self.eval_mode_value)

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
                          num_workers=num_workers,
                          collate_fn=self.collate)
    
    @staticmethod
    def collate(batch):
        imgs = []
        targets = []

        for (img, target) in batch:
            imgs.append(img)
            targets.append(target)

        return torch.stack(imgs, 0), targets


