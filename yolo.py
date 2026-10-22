import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledHead(nn.Module):
    def __init__(self,in_channels=256,num_classes=10,hidden=256):
            super().__init__()

            self.classification=nn.Sequential(
                  ConvBlock(in_channels,hidden,3,1,1),
                  ConvBlock(hidden,hidden,3,1,1),
                  nn.Conv2d(hidden,num_classes,1,1,0)
            )

            self.regression=nn.Sequential(
                              ConvBlock(in_channels,hidden,3,1,1),
                              ConvBlock(hidden,hidden,3,1,1),
                              nn.Conv2d(hidden,4,1,1,0)
                        )

    def forward(self,x):
          y1=self.classification(x)
          y2=self.regression(x)

          return y1,y2

class YOLOHead(nn.Module):
      def __init__(self, in_channels=256, num_classes=10):
        super().__init__()
        self.head_p2 = DecoupledHead(in_channels, num_classes)
        self.head_p3 = DecoupledHead(in_channels, num_classes)
        self.head_p4 = DecoupledHead(in_channels, num_classes)

      def forward(self, p2, p3, p4):
            cls2,reg2=self.head_p2(p2)
            cls3,reg3=self.head_p3(p3)
            cls4,reg4=self.head_p4(p4)
        
            return [cls2, cls3, cls4], [reg2, reg3, reg4]