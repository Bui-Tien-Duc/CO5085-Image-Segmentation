"""2D nested-skip UNet++ adaptation; single output, no deep supervision.
Reference: Zhou et al., IEEE TMI 2020, doi:10.1109/TMI.2019.2959609.
Uses the existing project's DoubleConv; not a full paper reproduction.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .unet import DoubleConv


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=11, base=16):
        super().__init__()
        widths = [base * 2**i for i in range(5)]
        self.pool = nn.MaxPool2d(2)
        self.blocks = nn.ModuleDict()
        for i in range(5):
            self.blocks[f'{i}_0'] = DoubleConv(
                in_channels if i == 0 else widths[i-1], widths[i])
        for j in range(1, 5):
            for i in range(5-j):
                self.blocks[f'{i}_{j}'] = DoubleConv(
                    j * widths[i] + widths[i+1], widths[i])
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, image):
        nodes = {(0, 0): self.blocks['0_0'](image)}
        for i in range(1, 5):
            nodes[i, 0] = self.blocks[f'{i}_0'](self.pool(nodes[i-1, 0]))
        for j in range(1, 5):
            for i in range(5-j):
                up = F.interpolate(nodes[i+1, j-1],
                    size=nodes[i, 0].shape[-2:], mode='bilinear', align_corners=False)
                merged = torch.cat([nodes[i, k] for k in range(j)] + [up], dim=1)
                nodes[i, j] = self.blocks[f'{i}_{j}'](merged)
        return self.head(nodes[0, 4])
