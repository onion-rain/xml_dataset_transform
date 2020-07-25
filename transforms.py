import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import utils as vutils
from PIL import Image


__all__ = ["HorizontalFlip", "CutOut", "CutOut", "CutMix", "Mosai"]

class HorizontalFlip(object):

    def img_transform(self, raw_img_path, new_img_path):
        img = transforms.ToTensor()(Image.open(raw_img_path))
        img = torch.flip(img, [-1])
        vutils.save_image(img, new_img_path)

    def label_transform(self, img_width, bbox):
        bbox[0] = img_width - bbox[0]
        bbox[2] = img_width - bbox[2]
        return bbox
        

class CutOut(object):
    """
    TODO 随机cut掉部分区域，填充0或随机噪声
    args:
        pad(int): 0为填充0，1为填充随机噪声
    """
    def __init__(self, pad=0):
        self.pad = pad

    def __call__(self, img, targets):

        return img, targets

    def __repr__(self):
        return self.__class__.__name__ + '(pad={0})'.format(self.pad)


# .88b  d88. d888888b db    db     d8888b. d888888b  .o88b. d888888b db    db d8888b. d88888b .d8888. 
# 88'YbdP`88   `88'   `8b  d8'     88  `8D   `88'   d8P  Y8 `~~88~~' 88    88 88  `8D 88'     88'  YP 
# 88  88  88    88     `8bd8'      88oodD'    88    8P         88    88    88 88oobY' 88ooooo `8bo.   
# 88  88  88    88     .dPYb.      88~~~      88    8b         88    88    88 88`8b   88~~~~~   `Y8b. 
# 88  88  88   .88.   .8P  Y8.     88        .88.   Y8b  d8    88    88b  d88 88 `88. 88.     db   8D 
# YP  YP  YP Y888888P YP    YP     88      Y888888P  `Y88P'    YP    ~Y8888P' 88   YD Y88888P `8888Y' 


class MixUp(object):
    """
    TODO 两张图片按比例混合
    args:
        p(float): 混合比例"""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, imgs, targets):

        return img, targets

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)

class CutMix(object):
    """
    TODO cutout填充其他图片"""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, imgs, targets):

        return img, targets

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)

class Mosai(object):
    """
    TODO 拼接多张图片"""
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, imgs, targets):

        return img, targets

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)
    