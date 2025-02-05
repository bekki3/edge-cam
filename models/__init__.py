import os
import re
import torch

from .ssdliteV2 import SSDLiteV2, build_ssdlite
from .mobilenetv2 import mobilenet_v2


def build_model(args,num=21, params=None,custom_class_num=None, pretrained=False):
    model = args.model.lower()

    if model.startswith('ssdlite'):
        return build_ssdlite(model, args, params, pretrained=pretrained,class_num=num)
    elif model.startswith('mobilenetv2'):
        return mobilenet_v2(custom_class_num=custom_class_num)

    raise Exception("unknown model %s" % args.model)


def load_model(model, source, optimizer=None, eval=0):
    if not os.path.isfile(source):
        raise Exception("can not open checkpoint %s" % source)
    checkpoint = torch.load(source, map_location = torch.device('cpu'))
    if eval==1:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
    return epoch, loss


def save_model(model,epoch,loss,optimizer, path='./checkpoints', postfix=None):
    if postfix:
        postfix = '_' + postfix
    else:
        postfix = ''

    target = os.path.join(path, f"{model.name}{postfix}_ep{epoch}_ls{loss:.3f}.pth")

    if not os.path.isdir(path):
        os.makedirs(path)

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, target)