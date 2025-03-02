# -*- coding: utf-8 -*-
# @Time    : 2024/5/27 19:08
# @Author  : lil louis
# @Location: Beijing
# @File    : utils.py

import os
from sklearn.model_selection import KFold
from dataset import FTDataset
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import torch
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
from calibration import calibration_front, calibration_profile
from PIL import Image
import cv2

saved_images = "./saved_image"

def get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                train_transform, val_transform,
                label_excel_dir, batch_size):
    labels_df = pd.read_excel(label_excel_dir)
    label_map = {row['Index'] : row['Tongue_Info'] for index, row in labels_df.iterrows()}
    train_dataset = FTDataset(img_root=train_img_dir, mask_root=train_mask_dir, label_map=label_map, transform=train_transform)
    val_dataset = FTDataset(img_root=val_img_dir, mask_root=val_mask_dir, label_map=label_map, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def train_fn(loader, model, loss_fn, optimizer, scaler, total_epoch, num_class, device):
    model.train()
    loop = tqdm(loader)
    total_loss, total_acc, total_iou, total_dice = 0.0, 0.0, 0.0, 0.0
    confusionmatrix_mean = np.zeros((num_class, num_class), dtype=np.int32)
    tongue_save = []
    for batch_index, ((img, mask, mask_hot), (patient_id, tongue_info)) in enumerate(loop):
        img = img.permute(0, 3, 1, 2).to(device).float()
        mask = mask.unsqueeze(1).to(device)  # mask [bs, 1, 512, 512]
        mask_hot = mask_hot.to(device)  # mask_hot [bs, num_class, 512, 512]
        with torch.cuda.amp.autocast():
            pred = model(img)  # pred [bs, num_class, 512, 512]
            loss = loss_fn(pred, mask_hot)

        # 梯度更新
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pred = torch.argmax(pred, dim=1).unsqueeze(1)  # pred [bs, 1, 512, 512]'
        if -1 not in patient_id:
            calibration_score, tongue_s = calibration_profile(pred, patient_id, tongue_info)
            tongue_save.append(tongue_s)

        # 保存图像
        # visual_save(pred, mask, patient_id, total_epoch, batch_index, device)
        # 准确率计算
        confusionmatrix = genConfusionMatrix(pred, mask, num_class)
        confusionmatrix_mean += confusionmatrix
        acc, iou, dice = cal_Matrix(confusionmatrix_mean, num_class)
        # 累加
        total_loss += loss.item()
        total_acc += acc
        total_dice += dice
        total_iou += iou
        loop.set_postfix(loss=loss.item())
    mean_acc, mean_dice, mean_iou, mean_loss = total_acc / len(loader), \
                                               total_dice/len(loader), \
                                               total_iou/len(loader), \
                                               total_loss / len(loader),
    #print(f"Train Epoch: {total_epoch}, Acc: {mean_acc}, Dice: {mean_dice}, mIOU: {mean_iou}")
    return (mean_acc, mean_dice, mean_iou, mean_loss), tongue_save


def val_check(loader, model, loss_fn, total_epoch, num_class, device):
    saved_path = "./saved_image/val"
    model.eval()
    loop = tqdm(loader)
    total_loss, total_acc, total_iou, total_dice = 0.0, 0.0, 0.0, 0.0
    total_calibration = [0, 0, 0, 0]
    confusionmatrix_mean = np.zeros((num_class, num_class), dtype=np.int32)
    tongue_save = []
    for batch_idx, ((img, mask, mask_hot), (patient_id, tongue_info)) in enumerate(loop):
        img = img.permute(0, 3, 1, 2).to(device).float()
        mask = mask.unsqueeze(1).to(device)
        mask_hot = mask_hot.to(device)
        with torch.cuda.amp.autocast():
            pred = model(img)
            loss = loss_fn(pred, mask_hot)
        pred = torch.argmax(pred, dim=1).unsqueeze(1)
        if -1 not in patient_id:
            calibration_score, tongue_s = calibration_profile(pred, patient_id, tongue_info)
            tongue_save.append(tongue_s)
            total_calibration = [x + y for x, y in zip(calibration_score, total_calibration)]
        #if total_epoch % 2 == 0:
            #save_img(pred, mask, total_epoch, batch_idx, saved_path)
        # 准确率计算
        confusionmatrix = genConfusionMatrix(pred, mask, num_class)
        confusionmatrix_mean += confusionmatrix
        acc, iou, dice = cal_Matrix(confusionmatrix_mean, num_class)
        total_loss += loss.item()
        total_acc += acc
        total_dice += dice
        total_iou += iou
    mean_acc, mean_dice, mean_iou, mean_loss = total_acc/len(loader), \
                                               total_dice/len(loader), \
                                               total_iou/len(loader), \
                                               total_loss/len(loader)
    mean_calibration = [x / len(loader) for x in total_calibration]
    #print(f"Val Epoch: {total_epoch}, Acc: {mean_acc}, Dice: {mean_dice}, mIOU: {mean_iou}")

    return(mean_acc, mean_dice, mean_iou, mean_loss), mean_calibration, tongue_save


def visual_save(pred, mask, patient_id_list, total_epoch, batch_index, device):
    pred = pred.squeeze().cpu().numpy()
    mask = mask.permute(0, 2, 3, 1).squeeze().cpu().numpy()  # 红色的标注图像尺寸应当为[256, 256]
    if len(patient_id_list) != pred.shape[0]:
        return
    for pred_index in range(pred.shape[0]):
        patient_id = patient_id_list[pred_index]
        out_path = f"./visualization/save/profile_dataset3_renew/1-Unet/{patient_id}_{total_epoch}_{batch_index}.png"

        pred_tmp = np.array(pred[pred_index])
        mask_tmp = np.array(mask[pred_index])

        pred_tmp = (pred_tmp * 255 / pred_tmp.max()).astype(np.uint8)
        mask_tmp = (mask_tmp * 255 / mask_tmp.max()).astype(np.uint8)

        # Ensure the images are binary
        _, pred_tmp = cv2.threshold(pred_tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask_tmp = cv2.threshold(mask_tmp, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

        edges_pred = cv2.Canny(pred_tmp, 50, 200)
        contours_pred, _ = cv2.findContours(edges_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_color = cv2.cvtColor(pred_tmp, cv2.COLOR_GRAY2BGR)
        # all plot
        cv2.drawContours(img_color, contours_pred, -1, (255, 0, 0), 2)
        # largest_contour_pred = get_largest_contour(contours_pred)
        # if largest_contour_pred is not None:
        #     cv2.drawContours(img_color, [largest_contour_pred], -1, (255, 0, 0), 2)

        edges_gt = cv2.Canny(mask_tmp, 50, 200)
        contours_gt, _ = cv2.findContours(edges_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_color, contours_gt, -1, (0, 0, 255), 1)
        Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)).save(out_path)
    return


def genConfusionMatrix(pred, label, num_class):
    pred, label = np.array(pred.cpu()), np.array(label.cpu())
    pred, label = pred.flatten(), label.flatten()
    mask = (label >= 0) & (label < num_class)
    label = num_class * label[mask] + pred[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusionmatrix = count.reshape(num_class, num_class)
    return confusionmatrix


def cal_Matrix(confusion_matrix, num_class):
    acc = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    assert len(confusion_matrix) == num_class
    iou = []
    for i in range(len(confusion_matrix)):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        iou_i = tp / (tp + fp + fn + 1e-8)
        iou.append(iou_i)
    mean_iou = np.mean(iou)

    dice = []
    for i in range(len(confusion_matrix)):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        dice_i = 2 * tp / (2 * tp + fp + fn + 1e-8)
        dice.append(dice_i)
    mean_dice = np.mean(dice)

    return acc, mean_iou, mean_dice


def save_img(pred, mask, total_epoch, batch_index, saved_path):
    print(f"==> Saving {total_epoch} pred & mask")
    pred = pred[0][0].cpu().numpy()
    mask = mask[0][0].cpu().numpy()
    #print(f"Saving First Batch Stage: {total_epoch}")
    #print(f"Saving Pred shape is {pred.shape}, Mask shape is {mask.shape}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pred)
    axes[0].set_title(f"Predicted in epoch:{total_epoch} Index:{batch_index}")

    axes[1].imshow(mask)
    axes[1].set_title(f"Mask in epoch:{total_epoch} Index:{batch_index}")
    plt.savefig(f"{saved_path}/pred_{total_epoch}_{batch_index}.png")
    plt.close()


def get_largest_contour(contours):
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

