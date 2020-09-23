import os
import torchvision.transforms as transforms
from PIL import Image
from torchvision import utils as vutils

root = "seg/train/"

seg_root = root + "SegmentationClassRaw/"
output_root = root + "NewSegmentationClassRaw/"

classed_path = {}
doing_path = []
doing_prefix = ''

def check_and_clear(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)
check_and_clear(output_root)

paths = os.listdir(seg_root)
for path in paths:
    prefix = "_".join(path.split("_")[:-1])
    if prefix in classed_path:
        classed_path[prefix].append(path)
    else:
        classed_path[prefix] = [path]
for key in classed_path.keys():
    img = None
    for path in classed_path[key]:
        if img is None:
            img = transforms.ToTensor()(Image.open(seg_root + path))
        else:
            img += transforms.ToTensor()(Image.open(seg_root + path))
    vutils.save_image(img, output_root + key + ".png")
