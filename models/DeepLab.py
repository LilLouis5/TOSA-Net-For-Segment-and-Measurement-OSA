# -*- coding: utf-8 -*-
# @Time    : 2024/3/20 8:45
# @Author  : lil louis
# @Location: Beijing
# @File    : DeepLab.py

import torch.nn as nn
from models.nets.mobilenetv2 import mobilenetv2
import torch
import torch.nn.functional as F
from models.nets.xception import xception

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        # Q1: 下面的三个参数是什么意思
        self.features = model.features[:-1]
        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                # Q2: 这下面的参数是什么意思？
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        # Q2: 这是什么封装方式
        classname = m.__class__.__name__
        # Q3: 下面的操作应该是修改步长和卷积块的大小
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # Q4: 下面这三步是怎么变换的？有什么作用
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        # 第一个分支，1x1卷积
        self.branch1 = nn.Sequential(
            # Q5: dilation的作用可能是起到膨胀卷积的作用
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 第二个分支, 3x3卷积，并且有6倍膨胀卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        #第三个分支, 3x3卷积，并且有12倍膨胀卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=12*rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        #第四个分支, 3x3卷积, 并且有18倍膨胀卷积
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=18*rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        #第五个分支, 全局池化
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()

        #五分支的分步卷积
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        #global_feature 进行的是第五个分支，全局池化操作
        #Q6: torch.mean的第二个参数是什么意思
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        # Q7: global_feature起到什么作用？
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        # Q8: F.interpolate的参数是什么作用？
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrain=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        # Q9: backbone是指模型中的哪个部分？ 经过backbone之后，feature map如何变化
        # 我认为backbone可能起到的作用是在一开始的encoder部分
        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrain=pretrain)
            # Q10: in_channels 和 low_level_channels 分别代表什么？
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrain)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError("Unsupported backbone - '{}', Use mobilenet, xception".format(backbone))

        # ASPP是五分支的包含膨胀卷积的卷积块
        self.aspp = ASPP(dim_in=in_channels, dim_out=256)

        # shortcut我认为是decoder当中的Low-Level Features的那个步骤
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # cat_conv是指在decoder中的concat 和 3x3 Conv的两部操作
        self.cat_conv = nn.Sequential(
            # 48是指decoder经过Low-Level Features走过来的一部分,
            # 256是指经过Encoder五分支卷积下来的feature
            nn.Conv2d(48+256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            #第二步 3x3conv 卷积
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # num_classes 和 transUnet是相同的都表示你总体有几个种类
        self.cls_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        #backbone的输出有两部分，low_level_features一部分是给Decoder的浅层特征，
        # x这部分是给Encoder的五分支任务的膨胀卷积部分
        low_level_features, x = self.backbone(x)
        #print(f"low_level_features shape is {low_level_features.shape}, x shape is {x.shape}")
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        # Q10: 这步我浅浅猜测是为了去resize
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x


if __name__ == "__main__":
    model = DeepLab(num_classes=1, backbone="mobilenet", pretrain=True, downsample_factor=16)
    x = torch.randn(size=(10, 3, 256, 256))
    preds = model(x)
    print(f"preds.shape: {preds.shape}")























