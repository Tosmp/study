import torch
import torch.nn as nn
import torch.nn.functional as F
from ..frelu import FReLU

class PSPHead(nn.Module):  
    def __init__(self, in_channels, num_classes):
        super(PSPHead, self).__init__()
        self.pyramid_pooling = PyramidPooling(in_channels=in_channels, pool_sizes=[6, 3, 2, 1])
        self.decode_feature = DecodePSPFeature(in_channels=in_channels*2, n_classes=num_classes)
        self.initialize()

    def forward(self, x):
        x = self.pyramid_pooling(x)
        #print("py_pool : {}".format(x.shape))
        output = self.decode_feature(x)
        #print("output : {}".format(output.shape))
        return output
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
class PSPHeadFReLU(nn.Module):  
    def __init__(self, in_channels, num_classes):
        super(PSPHeadFReLU, self).__init__()
        self.pyramid_pooling = PyramidPoolingFReLU(in_channels=in_channels, pool_sizes=[6, 3, 2, 1])
        self.decode_feature = DecodePSPFeatureFReLU(in_channels=in_channels*2, n_classes=num_classes)
        self.initialize()

    def forward(self, x):
        x = self.pyramid_pooling(x)
        #print("py_pool : {}".format(x.shape))
        output = self.decode_feature(x)
        #print("output : {}".format(output.shape))
        return output
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()                    

    
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPooling, self).__init__()

        # 各畳み込み層の出力チャネル数
        out_channels = int(in_channels / len(pool_sizes))

        # pool_sizes: [6, 3, 2, 1]
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        size = x.shape[-2:]
        out1 = self.cbr_1(self.avpool_1(x))
        out1 = F.interpolate(out1, size=size, mode="bilinear", align_corners=False)

        out2 = self.cbr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=size, mode="bilinear", align_corners=False)

        out3 = self.cbr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=size, mode="bilinear", align_corners=False)
        out4 = self.cbr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=size, mode="bilinear", align_corners=False)
        output = torch.cat([x, out1, out2, out3, out4], dim=1)

        return output
    
class PyramidPoolingFReLU(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPoolingFReLU, self).__init__()

        # 各畳み込み層の出力チャネル数
        out_channels = int(in_channels / len(pool_sizes))

        # pool_sizes: [6, 3, 2, 1]
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNormFRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2DBatchNormFRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNormFRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNormFRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        size = x.shape[-2:]
        out1 = self.cbr_1(self.avpool_1(x))
        out1 = F.interpolate(out1, size=size, mode="bilinear", align_corners=False)

        out2 = self.cbr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=size, mode="bilinear", align_corners=False)

        out3 = self.cbr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=size, mode="bilinear", align_corners=False)
        out4 = self.cbr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=size, mode="bilinear", align_corners=False)
        output = torch.cat([x, out1, out2, out3, out4], dim=1)

        return output

class DecodePSPFeature(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(DecodePSPFeature, self).__init__()

        self.cbr = conv2DBatchNormRelu(
            in_channels=in_channels, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        #self.cbr = conv2DBatchNormRelu(in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):        
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        return x
    
class DecodePSPFeatureFReLU(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(DecodePSPFeatureFReLU, self).__init__()

        self.cbr = conv2DBatchNormFRelu(
            in_channels=in_channels, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        #self.cbr = conv2DBatchNormRelu(in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        return x    

    
class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)
        return outputs
    
class conv2DBatchNormFRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNormFRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = FReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)
        return outputs