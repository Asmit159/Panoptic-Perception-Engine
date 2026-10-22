import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.layer2 = nn.BatchNorm2d(out_channels)
        self.layer3 = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.AdaptiveAvgPool2d((1, 1))
        self.layer2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.layer3 = nn.BatchNorm2d(out_channels)
        self.layer4 = nn.SiLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]
        
        y = self.layer1(x)
        y = self.layer4(self.layer3(self.layer2(y)))
        
        return F.interpolate(y, size=size, mode='bilinear', align_corners=False)


class SegHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        out_channels = 256

        self.branches = nn.ModuleList([
            ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ASPPConv(in_channels, out_channels, dilation=6),
            ASPPConv(in_channels, out_channels, dilation=12),
            ASPPConv(in_channels, out_channels, dilation=18),
            ASPPPooling(in_channels, out_channels)
        ])

        self.fusion = ConvBlock(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0)
        self.final_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        res = []
        for branch in self.branches:
            res.append(branch(x))

        y = torch.cat(res, dim=1)

        y = self.fusion(y)
        y = self.final_conv(y)

        return F.interpolate(y, scale_factor=8, mode='bilinear', align_corners=False)