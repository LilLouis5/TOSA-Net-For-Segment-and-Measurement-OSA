# -*- coding: utf-8 -*-
# @Time    : 2024/7/8 15:58
# @Author  : lil louis
# @Location: Beijing
# @File    : visualization.py

import torch
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

import matplotlib.pyplot as plt

from models.DE_Net import DE_Net
from models.TransUnet import TransUnet

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = ["back_ground", "tongue"]
num_class = len(num_classes)

# 在统一照片下改动 model_weight, output_path, model
model_weight = "./result/dataset1-result/norm/result1_Unet/model.pth"

IMAGE_SIZE = 512  # 你需要定义IMAGE_SIZE
TARGET_SIZE = 768  # dataset1
#TARGET_SIZE = 4032  # dataset3

output_path = "./visualization/save/final/1-Unet-Final.png"
img_path = "./visualization/img/1.png"
img_gt_path = "./visualization/img/1_GT.jpg"

def canny_plot(img, save_path):
    img = np.array(Image.fromarray(img.astype(np.uint8)))

    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)
    Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)).save(save_path)
    print("Save Success")
    return None

def main():
    # 用PIL加载图像并转换为NumPy数组
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)

    img_GT = Image.open(img_gt_path).convert("RGB")
    img_GT = np.array(img_GT)

    # 使用albumentations进行预处理
    transform = A.Compose(
        [
            # A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE,
            #               border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0], always_apply=True),
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            ToTensorV2()  # 转换为张量
        ]
    )

    augmented = transform(image=img)
    augmented_GT = transform(image=img_GT)
    img = augmented['image'].unsqueeze(0).float().to(device)  # 添加批次维度并移动到设备
    img_GT = augmented_GT['image']
    # 初始化模型

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_class
    ).to(device)

    # model = TransUnet(img_dim=IMAGE_SIZE,
    #                   in_channels=3,
    #                   out_channels=128,
    #                   head_num=4,
    #                   mlp_dim=512,
    #                   block_num=8,
    #                   patch_dim=16,
    #                   class_num=num_class).to(device)

    #model = DE_Net(in_channel=3, out_channel=num_class).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(model_weight))

    # 预测
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        pred = model(img)
    pred = torch.argmax(pred, dim=1)  # pred [bs, 512, 512]'
    pred = pred.squeeze().cpu().numpy()

    print("Prediction value range:", pred.min(), pred.max())

    # 归一化到0-255范围
    pred = (pred * 255 / pred.max()).astype(np.uint8)
    # 保存预测结果为图像
    pred_img = np.array(Image.fromarray(pred.astype(np.uint8)))
    img_GT = img_GT.permute(1, 2, 0)
    img_GT = np.array(img_GT)

    edges = cv2.Canny(pred_img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_color = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(img_color, contours, -1, (255, 0, 0), 2)  # 蓝色线条，线条宽度为2

    edges_gt = cv2.Canny(img_GT, 100, 200)
    contours_gt, _ = cv2.findContours(edges_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_color, contours_gt, -1, (0, 0, 255), 2)  # 蓝色线条，线条宽度为2

    Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)).save(output_path)
    return img_color


if __name__ == "__main__":
    prediction = main()
    print(prediction.shape)



