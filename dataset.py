# -*- coding: utf-8 -*-
# @Time    : 2024/5/27 17:06
# @Author  : lil louis
# @Location: Beijing
# @File    : dataset3.py

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os

def mask_to_onehot(mask, palette):
    mask = np.array(mask)
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour[0])
        class_map = np.all(equality, axis=-1)
        semantic_map.append(equality)
    semantic_map = np.stack(semantic_map).astype(np.float32)
    return semantic_map

def mask_to_onehot_RGB(mask, color_to_class):
    mask = np.array(mask.convert("RGB"))
    one_hot_mask = np.zeros((*mask.shape[:2], len(color_to_class)), dtype=np.uint8)
    for idx, color in enumerate(color_to_class):
        is_color = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
        one_hot_mask[:, :, idx] = is_color
    return one_hot_mask


class FTDataset(Dataset):
    def __init__(self, img_root, mask_root, label_map, transform=None):
        self.img_root = img_root
        self.mask_root = mask_root
        self.label_map = label_map
        self.transform = transform
        # self.img_files = [f for f in os.listdir(self.img_root)]
        # 下面这行代码的作用是找出OSA患者和非OSA患者
        self.img_files = [
            f for f in os.listdir(self.img_root)
            if float(self.label_map[int(f.split("-")[0])].split("_")[-1]) > 15
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        #palette = [[0], [1], [2], [3], [5]]  # palette: back_ground, washer-1, washer-3, front_tongue, hole
        palette = [[0], [1], [2], [4], [5]]  #profile

        patient_id = self.img_files[idx].split("-")[0]
        tongue_info = self.label_map[int(patient_id)].split("_")

        img_path = os.path.join(self.img_root, self.img_files[idx])

        #mask_path = os.path.join(self.mask_root, self.img_files[idx])  # dataset1
        #mask_path = os.path.join(self.mask_root, self.img_files[idx].replace(".JPG", ".png"))  # dataset2
        mask_path = os.path.join(self.mask_root, self.img_files[idx].replace(".JPG", "-mask.png"))  #dataset3 & dataset4
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path)).astype(np.uint8)
        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            img = augmentation["image"]
            mask = augmentation["mask"]
        mask_hot = mask_to_onehot(mask, palette)
        return (img, mask, mask_hot), (patient_id, tongue_info)



