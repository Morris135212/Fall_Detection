import torch
from torch.utils.data import Dataset
from .transforms import DEFAULT_TRANSFORMS
import cv2
import glob
import os
import re
import numpy as np
from PIL import Image

STRIDE = 1
DURATION = 10


def match(x):
    x = x.split("/")[-1]
    try:
        digits = re.search(r"[0-9]+", x).group()
        return int(digits)
    except:
        return 0


class CustomDataset(Dataset):
    def __init__(self, path, classes, transform=DEFAULT_TRANSFORMS):
        super(CustomDataset, self).__init__()
        self.path = path
        self.classes = classes
        self.transforms = transform
        self.__refersh__()

    def __len__(self):
        return len(self.annotations)

    def __refersh__(self):
        self.annotations = []
        root = self.path
        for i, cls in enumerate(self.classes):
            cls_root = os.path.join(root, cls)
            folders = os.listdir(cls_root)
            for folder in folders:
                try:
                    files = os.path.join(cls_root, folder)
                    imgs_path = glob.glob(f"{files}/*.jpg")
                    imgs_path = sorted(imgs_path, key=match)
                    for j in range(DURATION-1, len(imgs_path), STRIDE):
                        tmp = imgs_path[j-DURATION+1: j+1]
                        if len(tmp) != 10:
                            continue
                        self.annotations.append({
                            "paths": tmp,
                            "class_index": i,
                            "class_name": cls
                        })
                except:
                    print(f"Somthing wrong with reading {folder}")
                    continue

    def __getitem__(self, index):
        annotation = self.annotations[index]
        items = []
        for path in annotation["paths"]:
            img = Image.open(path)
            if self.transforms:
                img = self.transforms(img)
            items.append(img)
        imgs = torch.stack(items, dim=1)
        return imgs, annotation["class_index"]

