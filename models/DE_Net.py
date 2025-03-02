import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class HCDC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HCDC, self).__init__()
        self.diConv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.diConv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=2, bias=False)
        self.diConv3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=5, bias=False)
        self.diConv4 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=7, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.diConv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.diConv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.diConv3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.diConv4(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DE_Net(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, features=[64,128,256,512]):
        super(DE_Net, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channel, feature))
            in_channel = feature

        for feature in reversed(features):
            self.ups.append(nn.Conv2d(feature * 2, feature, kernel_size=1, stride=1, padding=0))
            self.ups.append(nn.ConvTranspose2d(feature, feature, kernel_size=3, stride=1, padding=1))
            self.ups.append(DoubleConv(in_channel=feature*2, out_channel=feature))

        self.HCDC_block = HCDC(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channel, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.HCDC_block(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)
            x = self.ups[idx+1](x)
            skip_connection = skip_connections[idx//3]
            if skip_connection.shape != x.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat = torch.cat((x, skip_connection), dim=1)
            x = self.ups[idx+2](concat)
        return self.final_conv(x)


def test():
    x = torch.randn((3, 3, 512, 512))
    model = DE_Net(in_channel=3, out_channel=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)


if __name__ == "__main__":
    test()