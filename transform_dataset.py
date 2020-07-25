from xml.dom.minidom import parse
import shutil
import os
from torchvision import utils as vutils
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import torch
from tqdm import tqdm
import random

raw_img_root = "/home/xueruini/onion_rain/pytorch/hw_auto_drive/raw/train-data/"
raw_label_root = "/home/xueruini/onion_rain/pytorch/hw_auto_drive/raw/object-detect-4Ks45fTUAnTRgL5kxIG/annotation/V006/annotations/"

new_dataset_root = "/home/xueruini/onion_rain/pytorch/hw_auto_drive/HorizontalFlip/"
# new_dataset_root = "/home/xueruini/onion_rain/pytorch/hw_auto_drive/test/"

prefix = new_dataset_root.split("/")[-2]+"_"

def check_and_clear(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)

def read_xml_root_node(xml_path):
    dom = parse(xml_path)
    root = dom.documentElement
    return root


def draw(img, detections, save_path):
    """
    args:
        img(3, h, w)
        detections: (x, y, x, y, cls), 单位为像素，
            注意：x, y为方框角坐标"""
    # Create plot
    img = img.transpose(0, -1).transpose(0, 1)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Draw bounding boxes and labels of detection
    if detections is not None:
        # n_cls_preds = len(detections)
        # bbox_colors = random.sample(colors, n_cls_preds)
        for idx, (x1, y1, x2, y2, cls) in enumerate(detections):
            box_w = x2 - x1
            box_h = y2 - y1
            # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # color = bbox_colors[idx]
            color = [0.9, 0.9, 0.2, 1]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1-20,
                fontsize=10,
                s=cls,
                color="black",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


if __name__ == "__main__":
    check_and_clear(new_dataset_root)

    dirs = os.listdir(raw_img_root)
    pbar = tqdm(
        dirs,
        desc="transforming",
        ncols=100
    )
    for dir in pbar:
        name = dir.split(".")[0]

        raw_img_path = raw_img_root+name+".jpg"
        new_img_path = new_dataset_root+prefix+name+".jpg"
        raw_label_path = raw_label_root+name+".xml"
        new_label_path = new_dataset_root+prefix+name+".xml"

        # copyfile(raw_img_path, new_img_path)
        img = transforms.ToTensor()(Image.open(raw_img_path))
        img = torch.flip(img, [-1])
        vutils.save_image(img, new_img_path)

        annotation_node = read_xml_root_node(raw_label_path)
        
        # 获取图片尺寸信息
        img_size = annotation_node.getElementsByTagName('size')[0]
        img_width = int(img_size.getElementsByTagName('width')[0].childNodes[0].nodeValue)
        img_height = int(img_size.getElementsByTagName('height')[0].childNodes[0].nodeValue)
        img_depth = int(img_size.getElementsByTagName('depth')[0].childNodes[0].nodeValue)

        annotation_node.getElementsByTagName('filename')[0].childNodes[0].nodeValue = new_img_path.split("/")[-1]

        objects = annotation_node.getElementsByTagName('object')
        bboxes = []
        for idx in range(len(objects)):
            object = objects[idx]
            bndbox = object.getElementsByTagName('bndbox')[0]

            # 根据transform修改bbox
            # HorizontalFlip
            raw_xmin = int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            raw_xmax = int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue = img_width - raw_xmin
            bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue = img_width - raw_xmax

        # # test draw
        #     # bbox以及label保存到list
        #     bbox = [int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue), 
        #             int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue), 
        #             int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue), 
        #             int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue), 
        #             object.getElementsByTagName('name')[0].childNodes[0].nodeValue]
        #     bboxes.append(bbox)
        # draw(img, bboxes, new_img_path)

        with open(new_label_path, "w+", encoding="utf-8") as f:
            annotation_node.writexml(f)
        # print()