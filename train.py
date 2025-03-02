# -*- coding: utf-8 -*-
# @Time    : 2024/5/27 19:18
# @Author  : lil louis
# @Location: Beijing
# @File    : train.py

from models.TransUnet import TransUnet
from models.DE_Net import DE_Net
from models.DeepLab import DeepLab

from models.TongueTrans import TongueTrans

import torch
from utils import get_loaders, train_fn, val_check, visual_save
import albumentations as A
import cv2
import torch.nn as nn
from torch import optim
import numpy as np
import os
import segmentation_models_pytorch as smp
import pandas as pd



device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
EPOCHS = 25
num_classes = ["back_ground", "washer-1", "washer-3", "front_tongue", "hole"]
#num_classes = ["back_ground", "washer-1", "washer-3", "profile_tongue", "hole"]
num_class = len(num_classes)
TARGET_SIZE = 4032  # dataset3
IMAGE_SIZE = 256  # TongueTrans
# IMAGE_SIZE = 512
saved_result_path = ""

LEARNING_RATE = 1e-4
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)

train_img_dir = "dataset3/train_img"
train_mask_dir = "dataset3/train_mask"
val_img_dir = "dataset3/val_img"
val_mask_dir = "dataset3/val_mask"

label_excel_dir = "./OSA.xlsx"
front_column_names = ['Index', 'Area', 'Area_label', 'Length', 'Length_label', 'Width', 'Width_label']
profile_column_names = ['Index', 'Area', 'Area_label', 'Length', 'Length_label', 'Thick', 'Thick_label', 'Curvature',
                        'Curvature_label']
visual_index = 1

def main():
    if not os.path.exists(saved_result_path):
        os.makedirs(saved_result_path)
    transform = A.Compose(
        [
            A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE,
                          border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0], always_apply=True),
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE)
        ]
    )
    # step1 导入模型以及训练参数
    # model = DE_Net(in_channel=3, out_channel=num_class).to(device)

    # model = smp.PSPNet(
    #     encoder_name="resnet34",
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=num_class
    # ).to(device)

    # model = TransUnet(img_dim=IMAGE_SIZE,
    #                   in_channels=3,
    #                   out_channels=128,
    #                   head_num=4,
    #                   mlp_dim=512,
    #                   block_num=8,
    #                   patch_dim=16,
    #                   class_num=num_class).to(device)

    model = TongueTrans(img_dim=IMAGE_SIZE,
                          in_channels=3,
                          out_channels=64,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          num_class=num_class).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0, 5.0, 8.0, 8.0], device=device))
    #loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0], device=device))
    optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    # step2 导入dataloader
    train_loader, val_loader = get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                                           train_transform=transform, val_transform=transform,
                                           label_excel_dir=label_excel_dir, batch_size=batch_size)
    TRAIN, VAL = [], []
    VAL_SAVE = []
    for index in range(EPOCHS):
        train, train_s = train_fn(train_loader, model, loss_fn, optimizer, scaler, index, num_class, device)
        val, val_c, val_s = val_check(val_loader, model, loss_fn, index, num_class, device)
        TRAIN.append(train), VAL.append(val)
        VAL_SAVE.append(val_s)
        print(f"训练集: {index}, Acc: {train[0]}, Dice: {train[1]}, mIOU: {train[2]}")
        print(f"验证集: {index}, Acc: {val[0]}, Dice: {val[1]}, mIOU: {val[2]}")
        # print(f"验证集: {index}, Area: {val_c[0]}, Length: {val_c[1]}, Width:{val_c[2]}")

        # val_s_df = [item for sublist in val_s for item in sublist]
        # val_s_df = pd.DataFrame(val_s_df, columns=column_names)
        # val_s_df.to_csv(f"{saved_result_path}/{index}_val_front.csv", index=False)

        train_s_df = [item for sublist in train_s for item in sublist]
        train_s_df = pd.DataFrame(train_s_df, columns=profile_column_names)
        train_s_df.to_csv(f"{saved_result_path}/{index}_train_front.csv", index=False)
    print("Train Finsh")
    #torch.save(model.state_dict(), f'{saved_result_path}/model.pth')
    #np.save(f"{saved_result_path}/train", TRAIN)
    #np.save(f"{saved_result_path}/val", VAL)


if __name__ == "__main__":
    main()
