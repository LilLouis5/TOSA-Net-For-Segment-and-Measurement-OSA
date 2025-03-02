# -*- coding: utf-8 -*-
# @Time    : 2024/6/24 11:11
# @Author  : lil louis
# @Location: Beijing
# @File    : calibration.py

import numpy as np
import math
from skimage.measure import find_contours
from scipy.interpolate import UnivariateSpline
import torch

def calibration_area(pred):
    washer3_area, front_tongue_area = 0, 0
    for i in range(len(pred)):
        for j in range(len(pred)):
            if pred[i][j] == 3:
                front_tongue_area += 1
            if pred[i][j] == 4 or pred[i][j] == 2:
                washer3_area += 1

    front_tongue_A = ((1.5 ** 2) * math.pi) * front_tongue_area / (washer3_area + 1e-8)
    return front_tongue_A


def calibration_length_width(pred):
    n = pred.shape[0]
    m = pred.shape[1]
    assert m == n
    f_lengths, f_widths = [0]*n, [0]*m
    w3_lengths, w3_widths = [0]*n, [0]*m

    # calculate for length
    for i in range(n):
        f_length, w1_length, w3_length = 0, 0, 0
        for j in range(m):
            if pred[i][j] == 3:
                f_length += 1
            if pred[i][j] == 4 or pred[i][j] == 2:
                w3_length += 1
        f_lengths[i], w3_lengths[i] = f_length, w3_length


    # calculate for width
    for j in range(m):
        f_width, w1_width, w3_width = 0, 0, 0
        for i in range(n):
            if pred[i][j] == 3:
                f_width += 1
            if pred[i][j] == 4 or pred[i][j] == 2:
                w3_width += 1
        f_widths[j], w3_widths[j] = f_width, w3_width

    f_length, w3_length = max(f_lengths), max(w3_lengths)
    f_width, w3_width = max(f_widths), max(w3_widths)

    f_length = 3 * f_length / (w3_length + 1e-8)
    f_width = 3 * f_width / (w3_width + 1e-8)
    return f_length, f_width


def curvature_splines(x, y, error=0.1):
    # 计算样条曲线并返回曲率
    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1/np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1/np.sqrt(std))

    dx = fx.derivative(1)(t)
    ddx = fx.derivative(2)(t)
    dy = fy.derivative(1)(t)
    ddy = fy.derivative(2)(t)
    curvature = (dx * ddy - dy * ddx) / (dx * dx + dy * dy)**1.5
    return t, curvature


def calibration_curvature(preds):
    # Ensure preds has a batch dimension
    if preds.ndim == 2:
        preds = np.expand_dims(preds, axis=0)

    preds = torch.tensor(preds)
    preds_binary = (preds == 3).float()
    preds_binary_np = preds_binary.cpu().numpy().astype(np.uint8)
    profile_curvature = []

    for i in range(preds_binary.shape[0]):
        single_image = preds_binary_np[i]

        if single_image.ndim != 2:
            print("Error: single_image is not a 2D array")
            profile_curvature.append(0)
            continue

        contours = find_contours(single_image, 0.5)
        if not contours:
            profile_curvature.append(0)
            # print("Error 侧面舌头有碎片的存在")
            continue

        tongue_contours = max(contours, key=lambda x: x.shape[0])
        x = tongue_contours[:, 1]
        y = tongue_contours[:, 0]
        t, curvature = curvature_splines(x, y)
        average_curvature = np.mean(np.abs(curvature))

        profile_curvature.append(average_curvature)
    curvature = profile_curvature[0]
    return curvature



def calibration_front(pred, patient_id, tongue_info):
    bs = pred.shape[0]
    area_score, length_score, width_score = 0, 0, 0
    tongue_info_save = []
    for i in range(bs):
        tongue_info_img = []
        pred_img = np.array(pred[i, 0, :, :].cpu())
        patient_id_save = patient_id[i]
        area = calibration_area(pred_img)
        length, width = calibration_length_width(pred_img)

        for tongue_index in range(7):
            tongue_info_img.append(float(tongue_info[tongue_index][i]))
        area_label, length_label, width_label = tongue_info_img[0], tongue_info_img[1], tongue_info_img[2]
        tongue_info_s = [patient_id_save, area, area_label, length, length_label, width, width_label]
        area_score = area_score + (1 - (abs(area - area_label)  / area_label))
        length_score = length_score + (1 - (abs(length - length_label) / length_label))
        width_score = width_score + (1 - (abs(width - width_label) / width_label))
        tongue_info_save.append(tongue_info_s)
    area_score, length_score, width_score = area_score/bs, length_score/bs, width_score/bs
    return [area_score, length_score, width_score], tongue_info_save



def calibration_profile(pred, patient_id, tongue_info):
    bs = pred.shape[0]
    area_score, length_score, thick_score, curvature_score = 0, 0, 0, 0
    tongue_info_save = []
    for i in range(bs):
        tongue_info_img = []
        pred_img = np.array(pred[i, 0, :, :].cpu())
        patient_id_save = patient_id[i]
        # np.save("./dataset4/pred_img", pred_img) # 7.1 当时为了确定palette所以存储
        area = calibration_area(pred_img)
        thick, length = calibration_length_width(pred_img)  #正面的侧面在这块是反着的
        cur = calibration_curvature(pred_img)
        cur = cur * 35
        for tongue_index in range(7):
            tongue_info_img.append(float(tongue_info[tongue_index][i]))
        area_label, length_label, thick_label, cur_label = tongue_info_img[3], tongue_info_img[4], tongue_info_img[5], tongue_info_img[6]
        tongue_info_s = [patient_id_save, area, area_label, length, length_label, thick, thick_label, cur, cur_label]
        area_score = area_score + (1 - (abs(area - area_label)  / area_label))
        length_score = length_score + (1 - (abs(length - length_label) / length_label))
        thick_score = thick_score + (1 - (abs(thick - thick_label) / thick_label))
        curvature_score = curvature_score + (1 - (abs(cur - cur_label) / cur_label))
        tongue_info_save.append(tongue_info_s)
    area_score, length_score, thick_score, curvature_score = area_score/bs, length_score/bs, thick_score/bs, curvature_score/bs
    return [area_score, length_score, thick_score, curvature_score], tongue_info_save

