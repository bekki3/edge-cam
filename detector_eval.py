import argparse
import torch
import os
from models import build_model, load_model
from helpers import DetectionEvaluator
from utils.txt_utils import read_txt

def prepare_model(args,num_classes):
    model = build_model(args, num=num_classes, pretrained=False)
    if not args.onnx:
        _=load_model(model, source=args.weight,eval=1)

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def main():
    parser = argparse.ArgumentParser(description='Detector Validation')

    parser.add_argument('--model', default='ssdlite320',choices=['ssdlite512','ssdlite320','ssdlite416','ssdlite640'],
                        help='Detector model name')
    parser.add_argument('--dataset_root', default='./PASCAL_VOC_Dataset/PASCAL_VOC_2007',type=str,
                        help='Dataset root directory path')
    parser.add_argument('--dataset_domains', default='test', type=str,
                        help='Dataset domains')
    parser.add_argument('--class_path', default='./PASCAL_VOC.txt', type=str,
                        help='class_label txt file directory')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--weight', default=None,
                        help='Weight file path')
    parser.add_argument('--onnx', default=None,
                        help='onnx file path -> use onnxruntime library')
    parser.add_argument('--th_iou', default=0.5, type=float,
                        help='IOU Threshold')
    parser.add_argument('--th_conf', default=0.05, type=float,
                        help='Confidence Threshold')
    parser.add_argument('--PR_curve', default=False, action='store_true',
                        help='save PR-Curve image in PR directory')
    parser.add_argument('--enable_letterbox', default=False, 
                        action='store_true',
                        help='Enable letterboxing image')
    parser.add_argument('--feature', default=False, 
                        action='store_true',
                        help='If set to True in ssdlite512 model, feature maps are extracted from 64*64 size')
    args = parser.parse_args()

    #check class_path
    if not os.path.isfile(args.class_path):
        raise Exception("There is no label data") 
            
    # load weight
    if args.onnx:
        if (os.path.isfile(args.onnx)):
            args.batch_size = 1
            pass
        else:
            raise RuntimeError('You must enter onnx path')
    elif not args.weight:
        if(os.path.isfile('checkpoints/' + args.model + '_obj_latest.pth')):
            args.weight = 'checkpoints/' + args.model + '_obj_latest.pth'
        else:
            raise RuntimeError('You must enter weight path')

    class_names = read_txt(args.class_path)
    num_classes = len(class_names)
    # prepare model
    model = prepare_model(args, num_classes)

    # validate dataset & print result
    evaluator = DetectionEvaluator(args, model)
    evaluator()


if __name__ == "__main__":
    main()
