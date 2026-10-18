import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=11, base=16):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = DoubleConv(in_channels, base)
        self.e2 = DoubleConv(base, base*2)
        self.e3 = DoubleConv(base*2, base*4)
        self.e4 = DoubleConv(base*4, base*8)
        self.bridge = DoubleConv(base*8, base*16)
        self.d4 = DoubleConv(base*16+base*8, base*8)
        self.d3 = DoubleConv(base*8+base*4, base*4)
        self.d2 = DoubleConv(base*4+base*2, base*2)
        self.d1 = DoubleConv(base*2+base, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def up(self, x, skip, block):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return block(torch.cat([skip, x], dim=1))

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        x = self.bridge(self.pool(s4))
        x = self.up(x, s4, self.d4)
        x = self.up(x, s3, self.d3)
        x = self.up(x, s2, self.d2)
        x = self.up(x, s1, self.d1)
        return self.head(x)
