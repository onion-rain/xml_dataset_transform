import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import utils as vutils
from PIL import Image
import random
import cv2

from torchtoolbox.transform import Cutout

__all__ = ["HorizontalFlip", "CutOut", "CutOut", "MixUp", "CutMix", "Mosai"]

class HorizontalFlip(object):
    """随机水平翻转"""
    def img_transform(self, raw_img_path, new_img_path):
        img = transforms.ToTensor()(Image.open(raw_img_path))
        img = torch.flip(img, [-1])
        vutils.save_image(img, new_img_path)
        return img

    def img_transform_2(self, raw_img_path, new_img_path, bboxes):
        pass

    def label_transform(self, img_width, bbox):
        bbox[0] = img_width - bbox[0]
        bbox[2] = img_width - bbox[2]
        return bbox
        
def _is_numpy_image(img):
    return img.ndim in {2, 3}

def cutout(img, i, j, h, w, v, inplace=False):
    """ Erase the CV Image with given value.

    Args:
        img (Tensor Image): Tensor image of size (C, H, W) to be erased
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v: Erasing value.
        inplace(bool, optional): For in-place operations. By default is set False.

    Returns:
        CV Image: Cutout image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    if not inplace:
        img = img.copy()

    img[i:i + h, j:j + w, :] = v
    return img


class CutOut(object):
    """
    TODO 随机cut掉部分区域，填充0或随机噪声
    args:
        pad(int): 0为填充0，1为填充随机噪声
    """
    def __init__(self, pixel_level=True, value=(0, 255)):
        self.pixel_level = pixel_level
        self.value = value

    def img_transform(self, raw_img_path, new_img_path):
        pass

    def img_transform_2(self, raw_img_path, new_img_path, bboxes):
        """置label_transform之后之后"""
        img = cv2.imread(raw_img_path)
        img_h = img.shape[0]
        img_w = img.shape[1]
        img_c = img.shape[2]
        # bbox_sizes = []
        for bbox in bboxes:  # bbox[xyxy]
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]

            pad_w = random.randint(int(bbox_w/3), int(bbox_w*4/5))
            pad_h = random.randint(int(bbox_h/3), int(bbox_h*4/5))
            
            pad_left_min = np.clip(bbox[0]-pad_w+int(bbox_w*0.4), 0, img_w-pad_w)
            pad_top_min  = np.clip(bbox[1]-pad_h+int(bbox_h*0.4), 0, img_h-pad_h)
            pad_left_max = np.clip(bbox[2]-int(bbox_w*0.4), 0, img_w-pad_w)
            pad_top_max  = np.clip(bbox[3]-int(bbox_h*0.4), 0, img_h-pad_h)

            left = random.randint(pad_left_min, pad_left_max)
            top = random.randint(pad_top_min, pad_top_max)
            # bbox_sizes.append([bbox_w, bbox_h])

            if self.pixel_level:
                pad = np.random.randint(*self.value, size=(pad_h, pad_w, img_c))
            else:
                pad = random.randint(*self.value)

            cutout(img, top, left, pad_h, pad_w, pad, inplace=True)
            
        cv2.imwrite(new_img_path, img)
        return img

        
class Cutout(object):
    """Random erase the given CV Image.

    It has been proposed in
    `Improved Regularization of Convolutional Neural Networks with Cutout`.
    `https://arxiv.org/pdf/1708.04552.pdf`


    Arguments:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        scale: range of proportion of erased area against input image.
        ratio: range of aspect ratio of erased area.
        pixel_level (bool): filling one number or not. Default value is False
    """

    def __init__(self, p=0.5, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4),
                 value=(0, 255), pixel_level=False, inplace=False):
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.pixel_level = pixel_level
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio):

        img_h, img_w, img_c = img.shape

        s = random.uniform(*scale)
        # if you img_h != img_w you may need this.
        # r_1 = max(r_1, (img_h*s)/img_w)
        # r_2 = min(r_2, img_h / (img_w*s))
        r = random.uniform(*ratio)
        s = s * img_h * img_w
        w = int(math.sqrt(s / r))
        h = int(math.sqrt(s * r))
        left = random.randint(0, img_w - w)
        top = random.randint(0, img_h - h)

        return left, top, h, w, img_c

    def __call__(self, img):
        if random.random() < self.p:
            left, top, h, w, ch = self.get_params(img, self.scale, self.ratio)
            if self.pixel_level:
                c = np.random.randint(*self.value, size=(h, w, ch))
            else:
                c = random.randint(*self.value)
            return F.cutout(img, top, left, h, w, c, self.inplace)
        return img



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
    def __init__(self, p=0.5):
        self.p = p

    def img_transform(self, raw_img1_path, raw_img2_path, new_img_path):
        img1 = transforms.ToTensor()(Image.open(raw_img1_path))
        img2 = transforms.ToTensor()(Image.open(raw_img2_path))
        new_img = img1*self.p + img2*self.p
        vutils.save_image(new_img, new_img_path)
        return new_img
    

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
    