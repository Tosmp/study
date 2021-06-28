from torch import nn
from ..frelu import FReLU



class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels, inter_channels=None):
        if inter_channels == None:
            inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)
        
class FCNHeadFReLU(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            FReLU(inter_channels),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHeadFReLU, self).__init__(*layers)
        
class FCNHead_FPN(nn.Sequential):
    def __init__(self, in_channels, channels, inter_channels=None):
        if inter_channels == None:
            inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead_FPN, self).__init__(*layers)