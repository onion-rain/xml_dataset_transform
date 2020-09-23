from xml.dom.minidom import parse
import shutil
import os
from PIL import Image
import torch
from tqdm import tqdm
import random
import copy

from test import *
from transforms import *

root = "raw4/"

raw_img_root = root + "raw_dataset/"
raw_label_root = root + "raw_dataset/"

# new_dataset_root = root + "Mosaic/"
new_dataset_root = "test/"

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
    return dom, root

def do_it(t, raw_img_root, raw_label_root, new_dataset_root, prefix, draw_flag=False):
    dirs = os.listdir(raw_img_root)
    dirs = [dir for dir in dirs if dir.endswith(".jpg")]
    dirs1 = dirs[:int(len(dirs)*1/4)-1]
    dirs2 = dirs[int(len(dirs)*1/4):int(len(dirs)*2/4)-1]
    dirs3 = dirs[int(len(dirs)*2/4):int(len(dirs)*3/4)-1]
    dirs4 = dirs[int(len(dirs)*3/4):]
    pbar = tqdm(
        range(min(len(dirs1), len(dirs2), len(dirs3), len(dirs4))),
        desc="transforming",
        ncols=100
    )
    for idx in pbar:
        name1 = dirs1[idx].split(".")[0]
        name2 = dirs2[idx].split(".")[0]
        name3 = dirs3[idx].split(".")[0]
        name4 = dirs4[idx].split(".")[0]

        raw_img1_path = raw_img_root+name1+".jpg"
        raw_label1_path = raw_label_root+name1+".xml"
        
        raw_img2_path = raw_img_root+name2+".jpg"
        raw_label2_path = raw_label_root+name2+".xml"
        
        raw_img3_path = raw_img_root+name3+".jpg"
        raw_label3_path = raw_label_root+name3+".xml"
        
        raw_img4_path = raw_img_root+name4+".jpg"
        raw_label4_path = raw_label_root+name4+".xml"

        new_img_path   = new_dataset_root+prefix+name1+"_"+name2+"_"+name3+"_"+name4+".jpg"
        new_label_path = new_dataset_root+prefix+name1+"_"+name2+"_"+name3+"_"+name4+".xml"

        img = t.img_transform(raw_img1_path, raw_img2_path, raw_img3_path, raw_img4_path, new_img_path)

        annotation_nodes = [None]*4
        _, annotation_nodes[0] = read_xml_root_node(raw_label1_path)
        _, annotation_nodes[1] = read_xml_root_node(raw_label2_path)
        _, annotation_nodes[2] = read_xml_root_node(raw_label3_path)
        _, annotation_nodes[3] = read_xml_root_node(raw_label4_path)

        # 获得清空object的root node
        annotation_new, annotation_node_new = read_xml_root_node(raw_label1_path)
        objects = annotation_node_new.getElementsByTagName('object')
        for object in objects:
            annotation_node_new.removeChild(object)
        object_frame = object
        # 改filename属性
        annotation_node_new.getElementsByTagName('filename')[0].childNodes[0].nodeValue = new_img_path.split("/")[-1]
        
        bboxes = [[]]*4
        for img_idx in range(4):
            annotation_node = annotation_nodes[img_idx]
            objects = annotation_node.getElementsByTagName('object')
            for bbox_idx in range(len(objects)):
                object = objects[bbox_idx]
                bndbox = object.getElementsByTagName('bndbox')[0]
                name = object.getElementsByTagName('name')[0].childNodes[0].nodeValue

                # bbox以及label保存到list
                bbox = [int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue), 
                        int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue), 
                        int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue), 
                        int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue),
                        name]
                bboxes[img_idx].append(bbox)

        bboxes = t.label_transform(bboxes)

        # 写入annotation_node_new
        # objects = annotation_node_new.createElement('object')
        # objects = annotation_new.createElement('object')

        for bbox in bboxes:
            annotation_node_new.appendChild(copy.copy(object_frame))
            bndbox = object_frame.getElementsByTagName('bndbox')[0]
            bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue = min(bbox[0], bbox[2])
            bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue = min(bbox[1], bbox[3])
            bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue = max(bbox[0], bbox[2])
            bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue = max(bbox[1], bbox[3])

        
        with open(new_label_path, "w+", encoding="utf-8") as f:
            annotation_node_new.writexml(f)

        # test draw
        if draw_flag:
            draw(img, bboxes, new_img_path)

if __name__ == "__main__":
    t = Mosaic()

    check_and_clear(new_dataset_root)

    # do_it(t, raw_img_root, raw_label_root, new_dataset_root, prefix, draw_flag=False)
    do_it(t, raw_img_root, raw_label_root, new_dataset_root, prefix, draw_flag=True)
