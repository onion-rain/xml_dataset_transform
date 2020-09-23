from xml.dom.minidom import parse
import shutil
import os
from PIL import Image
import torch
from tqdm import tqdm
import random

from test import *
from transforms import *

root = "raw4/"

raw_img_root = root + "sidewalk/"
raw_label_root = root + "sidewalk/"

new_dataset_root = root + "HorizontalFlip/"
# new_dataset_root = "/home/xueruini/onion_rain/pytorch/xml_dataset_transform/test/"

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

def do_it(t, raw_img_root, raw_label_root, new_dataset_root, prefix, draw_flag=False):
    dirs = os.listdir(raw_img_root)
    dirs = [dir for dir in dirs if dir.endswith(".jpg")]
    # sample_ids = random.sample(dirs, 1000)
    sample_ids = dirs
    pbar = tqdm(
        sample_ids,
        desc="transforming",
        ncols=100
    )
    for dir in pbar:
        name = dir.split(".")[0]

        raw_img_path = raw_img_root+name+".jpg"
        new_img_path = new_dataset_root+prefix+name+".jpg"
        raw_label_path = raw_label_root+name+".xml"
        new_label_path = new_dataset_root+prefix+name+".xml"

        img = t.img_transform(raw_img_path, new_img_path)

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
            name = object.getElementsByTagName('name')[0].childNodes[0].nodeValue

            # bbox以及label保存到list
            bbox = [int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue), 
                    int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue), 
                    int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue), 
                    int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue),]
            # bboxes.append(bbox)

            # 根据transform修改bbox
            bbox = t.label_transform(img_width, bbox)
            bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue = min(bbox[0], bbox[2])
            bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue = min(bbox[1], bbox[3])
            bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue = max(bbox[0], bbox[2])
            bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue = max(bbox[1], bbox[3])

            # bbox以及label保存到list
            show_bbox = [int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue), 
                        int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue), 
                        int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue), 
                        int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue),
                        name]
            bboxes.append(show_bbox)

        with open(new_label_path, "w+", encoding="utf-8") as f:
            annotation_node.writexml(f)

        t.img_transform_2(raw_img_path, new_img_path, bboxes)

        # test draw
        if draw_flag:
            draw(img, bboxes, new_img_path)

if __name__ == "__main__":
    t = HorizontalFlip()

    check_and_clear(new_dataset_root)

    do_it(t, raw_img_root, raw_label_root, new_dataset_root, prefix, draw_flag=True)
