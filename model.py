import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding=-1):
        super().__init__()

        if padding==-1:
            padding=kernel_size//2

        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False)
        self.bn=nn.BatchNorm2d(out_channels)
        self.ac=nn.SiLU(inplace=True)

    def forward(self,x):
        return self.ac(self.bn(self.conv(x)))


class BottleNeck(nn.Module):
    def __init__(self,in_channels,out_channels,shortcut=True):
        super().__init__()

        self.layer1=ConvBlock(in_channels,out_channels,1,1)
        self.layer2=ConvBlock(out_channels,out_channels,3,1)

        if(shortcut and in_channels==out_channels):
            self.add=True
        else:
            self.add=False

    def forward(self,x):
        if(self.add):
            return x+self.layer2(self.layer1(x))
        else:
            return self.layer2(self.layer1(x))

class CSPBlock(nn.Module):
    def __init__(self,in_channels,out_channels,num_bottlenecks,shortcut=True):
            super().__init__()

            self.hidden_channels=out_channels//2
            self.dense=ConvBlock(in_channels,self.hidden_channels,1,1)
            self.bypass=ConvBlock(in_channels,self.hidden_channels,1,1)

            
            self.features=nn.Sequential(*(BottleNeck(self.hidden_channels, self.hidden_channels, shortcut) for _ in range(num_bottlenecks)))

            self.last=ConvBlock(2*self.hidden_channels,out_channels,1,1)

    def forward(self,x):
            y1=self.features(self.dense(x))

            y2=self.bypass(x)
            y=torch.concat((y1,y2),dim=1)
            return self.last(y)

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden_channels = in_channels // 2
        
        self.cv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = ConvBlock(hidden_channels * 4, out_channels, kernel_size=1, stride=1)

        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class CSPDarknet(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem: 1/2 resolution
        self.stem = ConvBlock(3, 64, kernel_size=6, stride=2, padding=2)
        
        self.stage1 = nn.Sequential(ConvBlock(64, 128, 3, 2), CSPBlock(128, 128, num_bottlenecks=3))
        
        self.stage2 = nn.Sequential(ConvBlock(128, 256, 3, 2), CSPBlock(256, 256, num_bottlenecks=6))

        self.stage3 = nn.Sequential(ConvBlock(256, 512, 3, 2), CSPBlock(512, 512, num_bottlenecks=9))

        self.stage4 = nn.Sequential(ConvBlock(512, 1024, 3, 2), CSPBlock(1024, 1024, num_bottlenecks=3))

    def forward(self, x):
        x = self.stage1(self.stem(x))
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c2, c3, c4


# Inside model.py
class PanopticPerceptionEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CSPDarknet()
        self.sppf = SPPF(1024, 1024)
        out_channels = 256

        self.latlayer2 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.latlayer3 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.latlayer4 = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Wire in the Day 2 and Day 3 heads
        self.seg_head = SegHead(in_channels=out_channels, num_classes=1) 
        self.yolo_head = YOLOHead(in_channels=out_channels, num_classes=10) 
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        c2, c3, c4 = self.backbone(x) 
        c4 = self.sppf(c4)

        P4 = self.latlayer4(c4)
        p3_proj = self.latlayer3(c3)
        P3 = self._upsample_add(P4, p3_proj)
        P3 = self.smooth3(P3)
        
        p2_proj = self.latlayer2(c2)
        P2 = self._upsample_add(P3, p2_proj)
        P2 = self.smooth2(P2)

        # Execute the heads
        seg_mask = self.seg_head(P2)
        yolo_cls, yolo_reg = self.yolo_head(P2, P3, P4) 
        
        return seg_mask, yolo_cls, yolo_reg
