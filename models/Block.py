# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 18:39
# @Author  : lil louis
# @Location: Beijing
# @File    : Block.py

import torch.nn as nn
import torch
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=5)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=9)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels*5, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=5e-2),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.pool(x)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.final(x)
        return x


class Multi_Scale_Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, alpha):
        super().__init__()
        self.in_channels = in_channels
        self.alpha = alpha
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels)
        self.flatten = nn.Flatten()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.sig = nn.Sigmoid()
        self.aspp = ASPP(in_channels=3, out_channels=self.in_channels)

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_scale):
        x_down, x_res = x, x

        x = self.conv(x)
        x = self.conv(x)

        x_res = self.avg_pool(x_res)
        x_res = self.flatten(x_res)
        x_res = self.fc1(x_res)
        x_res = self.fc2(x_res)
        x_res = x_res.unsqueeze(-1).unsqueeze(-1)

        x_res = x_down * x_res
        x = x + x_res

        x = self.conv1(x)
        x = self.sig(x)
        x = x_down * x * self.alpha

        x_scale = self.aspp(x_scale)
        x_scale_down = x_scale
        x_scale = self.conv1(x_scale)
        x_scale = self.sig(x_scale)
        x_scale = x_scale_down * x_scale * (1-self.alpha)

        x = x + x_scale
        x = self.final_conv(x)
        return x

class Multi_Scale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels+3, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_scale):
        x = torch.cat((x, x_scale), dim=1)
        x = self.conv1(x)
        return x



def ECA(x, gamma=2, b=1):
    N, C, H, W = x.size()
    t = int(abs((math.log(C, 2 + b) / gamma)))
    k = t if t % 2 else t + 1

    avg_pool = nn.AdaptiveAvgPool2d(1)
    conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False).to(device)

    y = avg_pool(x)
    y = conv(y.squeeze(-1).transpose(-1, -2))
    y = y.transpose(-1, -2).unsqueeze(-1)

    return x * y.expand_as(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 64, 256, 256).to(device)
    x_scale = torch.randn(4, 3, 256, 256).to(device)
    test_block = Multi_Scale_Fusion(in_channels=64,
                                    out_channels=128,
                                    alpha=0.5).to(device)
    out = test_block(x, x_scale)
    print(out.shape)
