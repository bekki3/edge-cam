import torch
import random

from PIL import Image, ImageOps
from torchvision.transforms import functional as F

from utils.color import *

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list): list of transforms to compose.
    Example:
        >>> Compose([ToTensor(), Normalize()])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), torch.tensor(target)

class LetterBox:
    def __init__(self, fill=(123, 116, 103)):
        self.fill = fill

    def __call__(self, img, target):
        width, height = img.size
        if width != height:
            size_max = max(width, height)
            pad_w = size_max - width
            pad_h = size_max - height
            pad = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            for t in target:
                t[0] = (t[0] * width + pad[0]) / size_max
                t[1] = (t[1] * height + pad[1]) / size_max
                t[2] = (t[2] * width + pad[0]) / size_max
                t[3] = (t[3] * height + pad[1]) / size_max
            img = ImageOps.expand(img, pad, fill=self.fill)
        return img, target

class Resize:
    def __init__(self, size, interpolation=Image.LANCZOS):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, target):
        return img.resize(self.size, self.interpolation), target

class CropStub:
    @staticmethod
    def crop(img, offset, size):
        x1, y1 = offset
        x2 = x1 + size[0]
        y2 = y1 + size[1]
        return img.crop((x1, y1, x2, y2))

    @staticmethod
    def generate_target(target, offset, org_size, new_size):
        x_offset, y_offset = offset
        width, height = org_size
        new_w, new_h = new_size
        new_target = []
        for t in target:
            cx = (t[0] + t[2]) / 2. * width - x_offset
            cy = (t[1] + t[3]) / 2. * height - y_offset
            if cx < 0 or cx > new_w or cy < 0 or cy > new_h:
                continue
            x1 = max(t[0] * width - x_offset, 0)
            y1 = max(t[1] * height - y_offset, 0)
            x2 = min(t[2] * width - x_offset, new_w)
            y2 = min(t[3] * height - y_offset, new_h)
            cls = t[4]
            new_target.append([x1 / new_w,
                               y1 / new_h,
                               x2 / new_w,
                               y2 / new_h,
                               cls])
        return new_target

class RandomColorSpace:
    candidates = ('BGR', 'BRG', 'RGB', 'RBG', 'GBR', 'GRB')

    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target
        mode = random.choice(self.candidates)
        r, g, b = img.split()
        if mode == 'BGR':
            return Image.merge("RGB", (b, g, r)), target
        elif mode == 'BRG':
            return Image.merge("RGB", (b, r, g)), target
        elif mode == 'RBG':
            return Image.merge("RGB", (r, b, g)), target
        elif mode == 'RGB':
            return Image.merge("RGB", (r, g, b)), target
        elif mode == 'GBR':
            return Image.merge("RGB", (g, b, r)), target
        elif mode == 'GRB':
            return Image.merge("RGB", (g, r, b)), target
        else:
            raise Exception("Unknown color space mode")

class RandomHorizontalFlip:
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target
        img = ImageOps.mirror(img)
        for t in target:
            x1 = t[0]
            x2 = t[2]
            t[0] = 1.0 - x2
            t[2] = 1.0 - x1
        return img, target

class RandomDistort:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img, target):
        hsv = rgb2hsv(img2arr(img))
        if random.random() < self.p:
            delta = random.uniform(-0.1, 0.1)
            hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 1.0
        if random.random() < self.p:
            ratio = random.uniform(0.5, 1.5)
            hsv[:, :, 1] *= ratio
            hsv[:, :, 1] = hsv[:, :, 1].clip(0., 1.)
        if random.random() < self.p:
            ratio = random.uniform(0.5, 1.5)
            hsv[:, :, 2] *= ratio
            hsv[:, :, 2] = hsv[:, :, 2].clip(0., 255.)
        rgb = hsv2rgb(hsv)
        if random.random() < self.p:
            diff = random.uniform(-25.5, 25.5)
            rgb += diff
            rgb = rgb.clip(0., 255.)
        img = arr2img(rgb)
        return img, target

class RandomExpand:
    def __init__(self, p=0.5, ratio=(1.0, 4.0), fill=(123, 116, 103)):
        self.p = p
        self.ratio = ratio
        self.fill = fill

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target
        scale = random.uniform(self.ratio[0], self.ratio[1])
        width, height = img.size
        width_new = int(width * scale)
        height_new = int(height * scale)
        pad_l = random.randint(0, width_new - width)
        pad_t = random.randint(0, height_new - height)
        pad_r = width_new - width - pad_l
        pad_b = height_new - height - pad_t
        for t in target:
            t[0] = (t[0] * width + pad_l) / width_new
            t[1] = (t[1] * height + pad_t) / height_new
            t[2] = (t[2] * width + pad_l) / width_new
            t[3] = (t[3] * height + pad_t) / height_new
        pad = (pad_l, pad_t, pad_r, pad_b)
        img = ImageOps.expand(img, pad, fill=self.fill)
        return img, target

class RandomSamplePatch(CropStub):
    def __init__(self, max_trials=50):
        # Reduce the inner loop trial count to lessen CPU overhead.
        self.max_trials = max_trials

    def __call__(self, img, target):
        width, height = img.size
        for trial in range(7):
            target_iou = random.choice((0.0, 0.1, 0.3, 0.7, 0.9, 1.0))
            if target_iou == 1.0:
                return img, target
            for _ in range(self.max_trials):
                new_width = random.uniform(0.3, 1.0)
                new_height = random.uniform(0.3, 1.0)
                ratio = (new_width * width) / (new_height * height)
                if not 0.5 < ratio < 2.0:
                    continue
                rect = [random.uniform(0., 1. - new_width),
                        random.uniform(0., 1. - new_height),
                        0.,
                        0.]
                rect[2] = rect[0] + new_width
                rect[3] = rect[1] + new_height
                if target_iou != 0.0:
                    min_iou = self.calc_min_iou(target, rect)
                    if not target_iou < min_iou < (target_iou + 0.2):
                        continue
                offset = (int(rect[0] * width), int(rect[1] * height))
                new_size = (int((rect[2] - rect[0]) * width),
                            int((rect[3] - rect[1]) * height))
                target_ = self.generate_target(target, offset, (width, height), new_size)
                if not target_:
                    continue
                return self.crop(img, offset, new_size), target_
        return img, target

    @staticmethod
    def calc_min_iou(target, rect):
        crop_x1, crop_y1, crop_x2, crop_y2 = rect
        min_iou = 1.0
        for x1, y1, x2, y2, _ in target:
            x1_ = max(x1, crop_x1)
            y1_ = max(y1, crop_y1)
            x2_ = min(x2, crop_x2)
            y2_ = min(y2, crop_y2)
            if x2_ <= x1_ or y2_ <= y1_:
                continue
            w = max((min(x2, x2_) - max(x1, x1_)), 0.)
            h = max((min(y2, y2_) - max(y1, y1_)), 0.)
            intersect = w * h
            union = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - intersect
            iou = intersect / union
            if 0. < iou < min_iou:
                min_iou = iou
        return min_iou

class Normalize:
    def __init__(self, mean=[0.486, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        return F.normalize(img, self.mean, self.std), target
