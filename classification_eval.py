import argparse
import torch
import os
from models import build_model, load_model
from helpers import ClassificationEvaluator


def prepare_model(args,class_numes):
    model = build_model(args,custom_class_num=class_numes, pretrained=False)

    _=load_model(model,source=args.weight,eval=1)

    if torch.cuda.is_available():
        model = model.cuda()

    return model

def export_classificaton_model(size, model):
    w = size[0]
    h = size[1]

    x = torch.rand(1,3,h,w)
    if torch.cuda.is_available():
        x = x.cuda()
    filename = model.name + '.onnx'
    print('dumping network to %s' % filename)
    torch.onnx.export(model, x, filename, opset_version=9)


def main():
    parser = argparse.ArgumentParser(description='evaluate network')
    parser.add_argument('--model', default='mobilenetv2',choices=['mobilenetv2'],
                        help='Detector model name')
    parser.add_argument('--dataset_root', default='downloads',type=str,
                        help='Dataset root directory path')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=-1, type=int,
                        help='Number of workers used in dataloading')  
    # model parameter
    parser.add_argument('--topk', type=int,
                        default=1,
                        help='topk...')
    parser.add_argument('--mean', nargs=3, type=float,
                        default=(0.486, 0.456, 0.406),
                        help='mean for normalizing')
    parser.add_argument('--std', nargs=3, type=float,
                        default=(0.229, 0.224, 0.225),
                        help='std for normalizing')
    parser.add_argument('--weight', default=None,
                        help='Weight file path')
    parser.add_argument('--input_size', default=[320,320], type=int,nargs=2,
                        help='input size (width, height)')
    parser.add_argument('--export', default=False, 
                        action='store_true',
                        help='export onnx')
    args = parser.parse_args()

    # load weight
    if not args.weight:
        if(os.path.isfile('checkpoints/' + args.model + '_class_latest.pth')):
            args.weight = 'checkpoints/' + args.model + '_class_latest.pth'
        else:
            raise Exception('You must enter weight path')

    # dataset_root check
    if not os.path.isdir(args.dataset_root):
        raise Exception("There is no dataset_root dir") 
        
    # measure the class num
    class_names = os.walk(args.dataset_root).__next__()[1]

    # prepare model
    model = prepare_model(args,class_numes = len(class_names))
    if args.export:
        export_classificaton_model(args.input_size,model)
    # validate dataset & print result
    evaluator = ClassificationEvaluator(args, model)
    evaluator()


if __name__ == "__main__":
    main()
