from xml.dom.minidom import parse
import shutil
import os
from PIL import Image
import torch
from tqdm import tqdm
import random

from test import *
from transforms import *

root = "raw3/"

raw_img_root = root + "raw_dataset/"
raw_label_root = root + "raw_dataset/"

new_dataset_root = root + "MixUp/"
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
    dirs = os.listdir(raw_img_root) # 此处应筛选.jpg结尾，但为了对照实验准确，暂时将错就错吧
    # dirs = [dir for dir in dirs if dir.endswith(".jpg")]
    dirs1 = dirs[:int(len(dirs)/2)-1]
    dirs2 = dirs[int(len(dirs)/2):]
    pbar = tqdm(
        range(min(len(dirs1), len(dirs2))),
        desc="transforming",
        ncols=100
    )
    for idx in pbar:
        name1 = dirs1[idx].split(".")[0]
        name2 = dirs2[idx].split(".")[0]

        raw_img1_path = raw_img_root+name1+".jpg"
        raw_label1_path = raw_label_root+name1+".xml"
        
        raw_img2_path = raw_img_root+name2+".jpg"
        raw_label2_path = raw_label_root+name2+".xml"

        new_img_path = new_dataset_root+prefix+name1+"_"+name2+".jpg"
        new_label_path = new_dataset_root+prefix+name1+"_"+name2+".xml"

        img = t.img_transform(raw_img1_path, raw_img2_path, new_img_path)

        annotation_node1 = read_xml_root_node(raw_label1_path)
        annotation_node2 = read_xml_root_node(raw_label2_path)
        annotation_node_new = read_xml_root_node(raw_label1_path) # 

        # 改filename属性
        annotation_node_new.getElementsByTagName('filename')[0].childNodes[0].nodeValue = new_img_path.split("/")[-1]

        objects2 = annotation_node2.getElementsByTagName('object')
        for idx in range(len(objects2)):
            annotation_node_new.appendChild(objects2[idx])
            
        objects_new = annotation_node_new.getElementsByTagName('object')

        bboxes = []
        for idx in range(len(objects_new)):
            object = objects_new[idx]
            bndbox = object.getElementsByTagName('bndbox')[0]
            name = object.getElementsByTagName('name')[0].childNodes[0].nodeValue

            # bbox以及label保存到list
            show_bbox = [int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue), 
                    int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue), 
                    int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue), 
                    int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue),
                    name]
            bboxes.append(show_bbox)
        
        with open(new_label_path, "w+", encoding="utf-8") as f:
            annotation_node_new.writexml(f)

        # test draw
        if draw_flag:
            draw(img, bboxes, new_img_path)

if __name__ == "__main__":
    t = MixUp()

    check_and_clear(new_dataset_root)

    do_it(t, raw_img_root, raw_label_root, new_dataset_root, prefix, draw_flag=False)
    # do_it(t, raw_img_root, raw_label_root, new_dataset_root, prefix, draw_flag=True)
