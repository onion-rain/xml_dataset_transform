import torch
import numpy as np
import torch.nn.functional as F

"""以下transforms输入img均为tensor"""

__all__ = ["MyCompose", "HorizontalFlip", "DropSmallLabel", "CenterCrop", "Pad2Square"]

class MyCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> MyCompose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class HorizontalFlip(object):
    """随机水平翻转
    args:
        p(float): 翻转概率(default: 0.5)"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, targets):
        if np.random.random() < self.p:
            img = torch.flip(img, [-1])
            targets[:, 2] = 1 - targets[:, 2]
        return img, targets

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)

class DropSmallLabel(object):
    """丢弃过小的nodule
    args:
        w_thres(float): 宽度阈值(宽度/图片宽度)(default: 0.01)
        h_thres(float): 高度阈值(高度/图片高度)(default: 0.01)
        label_list(list): 参与筛选的label标号列表"""
    def __init__(self, w_thres=0.01, h_thres=0.01, label_list=[2]):
        self.w_thres = w_thres
        self.h_thres = h_thres
        self.label_list = label_list

    def __call__(self, img, targets):
        labels = []
        for label in targets:
            if label[1] not in self.label_list:
                labels.append(label.numpy()) # thyroid
            elif label[4] > self.w_thres and label[5] > self.h_thres:
                labels.append(label.numpy()) # nodule
            # else:
            #     from sample.draw_boxes import draw
            #     pic = img.transpose(0, -1).transpose(0, 1)
            #     detections = torch.cat((targets[:, 2:]*pic.shape[0], targets[:, 1:2]), 1)
            #     draw(pic, detections, "x.jpg")
            #     a = 1
        new_targets = torch.Tensor(labels)
        return img, new_targets

    def __repr__(self):
        return self.__class__.__name__ + '(w_thres={0}, h_thres={1}, label_list={2})'.format(self.w_thres, self.h_thres, self.label_list)

class CenterCrop(object):
    """
    按照比例将中部图像剪裁出来，边缘扔掉
    args:
        kp_w(float): 新图像w与旧图像w比例
        kp_h(float): 新图像h与旧图像h比例
    注：targets必须为比例
    """
    def __init__(self, kp_w=0.85, kp_h=0.6):
        self.kp_h = kp_h
        self.kp_w = kp_w

    def __call__(self, img, targets):
        h_o = img.size()[1]
        w_o = img.size()[2]
        h_n = int(round(h_o*self.kp_h))
        w_n = int(round(w_o*self.kp_w))

        h_start = int(round((h_o-h_n)/2))
        h_end = int(round(h_o+h_n)/2)

        w_start = int(round((w_o-w_n)/2))
        w_end = int(round(w_o+w_n)/2)
        # new_img = torch.zeros((3, h_n, w_n))
        new_img = img[:, h_start:h_end, w_start:w_end]
        
        targets[:, 2] = (2*targets[:, 2] -1 + self.kp_w) / 2 / self.kp_w
        targets[:, 3] = (2*targets[:, 3] -1 + self.kp_h) / 2 / self.kp_h
        targets[:, 4] = targets[:, 4] / self.kp_w
        targets[:, 5] = targets[:, 5] / self.kp_h

        return new_img, targets
    
    def __repr__(self):
        return self.__class__.__name__ + '(kp={0})'.format(self.kp)

class Pad2Square(object):
    """
    填充至正方形
    args:
        pad_value(int): 为长边填充量
    """
    def __init__(self, pad_value=0):
        self.pad_value = pad_value

    def __call__(self, img, targets):
        c, h, w = img.shape
        h_factor, w_factor = (h, w)
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=self.pad_value)
        _, padded_h, padded_w = img.shape

        boxes = targets[:, 1:]
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[0]
        y2 += pad[2]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h
        targets[:, 1:] = boxes
        return img, targets

    def __repr__(self):
        return self.__class__.__name__ + '(pad={0})'.format(self.pad)

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
    