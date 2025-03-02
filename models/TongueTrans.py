# -*- coding: utf-8 -*-
# @Time    : 2024/6/19 14:12
# @Author  : lil louis
# @Location: Beijing
# @File    : TongueTrans.py

import torch.nn as nn
from models.Block import Multi_Scale_Fusion, ECA, ASPP
import torch
from models.vit import ViT
from einops import rearrange

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = x + x_down
        x = self.relu(x)
        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_concat=None, x_res=None):
        x = self.upsample(x)

        if x_concat is not None and x_res is None:
            x = torch.cat((x, x_concat), dim=1)
        if x_concat is not None and x_res is not None:
            x = torch.cat((x, x_concat, x_res), dim=1)

        x = self.layer(x)
        return x



class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.multi1 = Multi_Scale_Fusion(out_channels, out_channels*2, alpha=0.5)
        self.multi2 = Multi_Scale_Fusion(out_channels*2, out_channels*4, alpha=0.5)
        self.multi3 = Multi_Scale_Fusion(out_channels*4, out_channels*8, alpha=0.5)

        self.encoder1 = EncoderBottleneck(out_channels*2, out_channels*2, stride=1)
        self.encoder2 = EncoderBottleneck(out_channels*4, out_channels*4, stride=1)
        self.encoder3 = EncoderBottleneck(out_channels*8, out_channels*8, stride=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels*8, out_channels*16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*16),
            nn.ReLU(inplace=True),
        )

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(img_dim=self.vit_img_dim,
                       in_channels=out_channels*16,
                       embedding_dim=out_channels*16,
                       head_num=head_num,
                       mlp_dim=mlp_dim,
                       block_num=block_num,
                       patch_dim=1,
                       classification=False)

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels*16, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(in_channels=512, out_channels=512)

    def forward(self, x):
        x_scale = x
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.maxpool(x1)
        x_scale = self.maxpool(x_scale)
        x2 = self.multi1(x2, x_scale)
        x2 = self.encoder1(x2)

        x3 = self.maxpool(x2)
        x_scale = self.maxpool(x_scale)
        x3 = self.multi2(x3, x_scale)
        x3 = self.encoder2(x3)

        x4 = self.maxpool(x3)
        x_scale = self.maxpool(x_scale)
        x4 = self.multi3(x4, x_scale)
        x4 = self.encoder3(x4)

        x = self.conv2(x4)
        x = self.vit(x)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x = self.conv3(x)
        x = self.aspp(x)
        return x, x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, out_channels, num_class):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels*16, out_channels*4)
        self.decoder2 = DecoderBottleneck(out_channels*8, out_channels*2)
        self.decoder3 = DecoderBottleneck(out_channels*6, out_channels)
        self.decoder4 = DecoderBottleneck(out_channels*3, int(out_channels // 2))

        self.decoder1_res = DecoderBottleneck(out_channels*4, out_channels*2, scale_factor=4)
        self.decoder2_res = DecoderBottleneck(out_channels*2, out_channels, scale_factor=4)

        self.conv1 = nn.Conv2d(int(out_channels // 2), num_class, kernel_size=1)

    def forward(self, x, x1, x2, x3, x4):
        x = self.decoder1(x, x4)           # x [512, 16, 16]  x4 [512, 32, 32]   -> x [256, 32, 32]
        x_res_1 = self.decoder1_res(x)     # x [256, 32, 32] ->  x_res_1 [128, 128, 128]
        x = self.decoder2(x, x3)           # x [256, 32, 32]  x3 [256, 64, 64]   -> x [128, 64, 64]
        x_res_2 = self.decoder2_res(x)     # x [128, 64, 64] -> x_res_2 [64, 256, 256]
        x = self.decoder3(x, x2, x_res_1)  # x [128, 64, 64]  x2 [128, 128, 128] -> x [64, 128, 128]
        x = self.decoder4(x, x1, x_res_2)  # x [64, 128, 128] x1 [64, 256, 256]  -> x [32, 256, 256]
        x = self.conv1(x)
        return x


class TongueTrans(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, num_class):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim)
        self.decoder = Decoder(out_channels, num_class)

    def forward(self, x):
        x, x1, x2, x3, x4 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3, x4)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 3, 256, 256).to(device)
    model = TongueTrans(img_dim=256,
                        in_channels=3,
                        out_channels=64,
                        head_num=4,
                        mlp_dim=512,
                        block_num=8,
                        patch_dim=16,
                        num_class=2).to(device)
    pred = model(x)
    print(pred.shape)