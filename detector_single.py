import argparse
import torch
from PIL import Image
import os
from models import build_model, load_model
from nn import DetectPostProcess
from utils.box_util import draw_object_box
from transforms import detector_transforms as transforms
from utils.txt_utils import read_txt
from ctypes import c_ubyte
from onnxsim import simplify
import onnx

def read_image(source):
    return Image.open(source)


def prepare_input(img, size, enable_letterbox):

    t = []

    if enable_letterbox:
        t.append(transforms.LetterBox())

    t.append(transforms.Resize(size))

    t = transforms.Compose(t)

    img, _ = t(img, [])

    t = transforms.Compose((transforms.ToTensor(),
                            transforms.Normalize()))

    tensor, _ = t(img, [])

    return img, tensor.unsqueeze(0)


def prepare_model(args, weight,num_classes):
    # build model
    model = build_model(args, num=num_classes,pretrained=False)
    post_process = DetectPostProcess(model.get_anchor_box(),
                                     args.th_conf,
                                     args.th_iou)

    # load weight
    _ = load_model(model, source=weight, eval=1)

    # transfer to GPU if possible
    if torch.cuda.is_available():
        model = model.cuda()

    return model, post_process


def inference(model, post_process, img, enable_letterbox):
    size = model.get_input_size()

    # prepare input
    img, x = prepare_input(img, size, enable_letterbox)

    # inference -> postprocess(softmax->nms->...)
    if torch.cuda.is_available():
        x = x.cuda()

    return img, post_process(model(x))


def single_run(model, post_process, x, class_names, 
               th_iou, th_conf, enable_letterbox):

    # change to evaluation mode
    model.eval()

    # inference image
    img = read_image(x)

    with torch.no_grad():
        img, results = inference(model, 
                                 post_process, 
                                 img, 
                                 enable_letterbox)
    
    # print results
    objs = []
    for _cls, _objs in enumerate(results[0]):
        if not _objs:
            continue
    
        label = class_names[_cls]
    
        for _obj in _objs:
            _obj.append(label)
            objs.append(_obj)
    
    img = draw_object_box(img, objs)
    img.show()


def export_model(model):
    size = model.get_input_size()

    w, h =size

    x = torch.rand(1,3,h,w)

    if torch.cuda.is_available():
        x = x.cuda()

    filename = model.name + '.onnx'
    print('dumping network to %s' % filename)

    torch.onnx.export(model, x, filename, opset_version=9)
    
    model_onnx = onnx.load(filename)
    onnx.checker.check_model(model_onnx)
    model_onnx, check = simplify(model_onnx)
    onnx.save(model_onnx, filename)
    
    filename = model.name + '.anchor'
    print('dumping anchor to %s' % filename)

    anchor = model.get_anchor_box().get_anchor()
    with open(filename, 'wb') as f:
        f.write(to_bytes(anchor))


def to_bytes(tensor):
    tensor = tensor * 256.0
    tensor = tensor.round().clamp(0., 255.).view(-1).int().tolist()

    t = c_ubyte * len(tensor)
    buf = t()

    for k, v in enumerate(tensor):
        buf[k] = v

    return buf


def main():
    parser = argparse.ArgumentParser(description='Detector Single Test')

    parser.add_argument('inputs', type=str, nargs='*',
                        help='Input image path')
    parser.add_argument('--model', default='ssdlite320', choices=['ssdlite320','ssdlite512','ssdlite416','ssdlite640'],
                        help='Detector model name')
    parser.add_argument('--class_path', default='./PASCAL_VOC.txt', type=str,
                        help='class_label txt file directory')
    parser.add_argument('--weight', default=None,
                        help='Weight file path')
    parser.add_argument('--th_conf', default=0.5, type=float,
                        help='Confidence Threshold')
    parser.add_argument('--th_iou', default=0.5, type=float,
                        help='IOU Threshold')
    parser.add_argument('--enable_letterbox', default=False, 
                        action='store_true',
                        help='Enable letterboxing image')
    parser.add_argument('--export', default=False,
                        action='store_true',
                        help='dump network to onnx format')
    parser.add_argument('--feature', default=False, 
                        action='store_true',
                        help='If set to True in ssdlite512 model, feature maps are extracted from 64*64 size')
    args = parser.parse_args()


    #check class_path
    if not os.path.isfile(args.class_path):
        raise Exception("There is no label data") 

    #check weight
    if not os.path.isfile(args.weight):
        raise Exception("There is no weight file") 

    # dataset
    class_names = read_txt(args.class_path)
    num_classes = len(class_names)

    # load weight
    if args.weight:
        weight = args.weight
    else:
        weight = 'checkpoints/' + args.model + '_obj_latest.pth'
    
    model, post_process = prepare_model(args, weight,num_classes)

    if args.export:
        export_model(model)

    for x in args.inputs:
        single_run(model, post_process, x, class_names, 
                   args.th_iou, args.th_conf, args.enable_letterbox)


if __name__ == "__main__":
    main()

