from xml.dom.minidom import parse
import shutil
import os
from PIL import Image
import torch
from tqdm import tqdm
import random

from test import *
from transforms import *

raw_img_root = "/home/xueruini/onion_rain/pytorch/xml_dataset_transform/raw/train-data/"
raw_label_root = "/home/xueruini/onion_rain/pytorch/xml_dataset_transform/raw/object-detect-4Ks45fTUAnTRgL5kxIG/annotation/V006/annotations/"

# new_dataset_root = "/home/xueruini/onion_rain/pytorch/xml_dataset_transform/HorizontalFlip/"
new_dataset_root = "/home/xueruini/onion_rain/pytorch/xml_dataset_transform/test/"

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

def do_it(t, raw_img_root, raw_label_root, new_dataset_root, prefix):
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

        t.img_transform(raw_img_path, new_img_path)

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

            # bbox以及label保存到list
            bbox = [int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue), 
                    int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue), 
                    int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue), 
                    int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue),]
            # bboxes.append(bbox)

            # 根据transform修改bbox
            bbox = t.label_transform(img_width, bbox)
            bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue = bbox[0]
            bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue = bbox[1]
            bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue = bbox[2]
            bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue = bbox[3]

        # # test draw
        # draw(img, bboxes, new_img_path)

        with open(new_label_path, "w+", encoding="utf-8") as f:
            annotation_node.writexml(f)

if __name__ == "__main__":
    t = HorizontalFlip()

    check_and_clear(new_dataset_root)

    do_it(t, raw_img_root, raw_label_root, new_dataset_root, prefix)
