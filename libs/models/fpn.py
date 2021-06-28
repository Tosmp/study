# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#from upsnet.operators.modules.distbatchnorm import BatchNorm2d
class FPN(nn.Module):

    def __init__(self, 
                 feature_dim, 
                 with_extra_level=False, 
                 with_norm='none', 
                 upsample_method='nearest'):
        super(FPN, self).__init__()
        self.feature_dim = feature_dim
        assert upsample_method in ['nearest', 'bilinear']

        def interpolate(input):
            return F.interpolate(input, 
                                 scale_factor=2, 
                                 mode=upsample_method, 
                                 align_corners=False if upsample_method == 'bilinear' else None)
        self.fpn_upsample = interpolate

        assert with_norm in ['group_norm', 'batch_norm', 'none']
        if with_extra_level:
            self.fpn_p6 = nn.MaxPool2d(kernel_size=1, stride=2)
        #if config.network.fpn_with_gap:
        #    self.fpn_gap = nn.Linear(2048, feature_dim)
        
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm

        if with_norm != 'none':
            self.fpn_p5_1x1 = nn.Sequential(*[nn.Conv2d(2048, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p4_1x1 = nn.Sequential(*[nn.Conv2d(1024, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p3_1x1 = nn.Sequential(*[nn.Conv2d(512, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p2_1x1 = nn.Sequential(*[nn.Conv2d(256, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p5 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p4 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p3 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p2 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            #self.fpn_p2 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            
        else:
            self.fpn_p5_1x1 = nn.Conv2d(2048, feature_dim, 1)
            self.fpn_p4_1x1 = nn.Conv2d(1024, feature_dim, 1)
            self.fpn_p3_1x1 = nn.Conv2d(512, feature_dim, 1)
            self.fpn_p2_1x1 = nn.Conv2d(256, feature_dim, 1)
            self.fpn_p5 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p4 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p3 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p2 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, res2, res3, res4, res5):
        fpn_p5_1x1 = self.fpn_p5_1x1(res5)
        fpn_p4_1x1 = self.fpn_p4_1x1(res4)
        fpn_p3_1x1 = self.fpn_p3_1x1(res3)
        if res3.shape[2]*2 != res2.shape[3]:
            res2 = F.interpolate(res2, size=(res3.shape[-2]*2, res3.shape[-1]*2), mode='bilinear', align_corners=False) 
        fpn_p2_1x1 = self.fpn_p2_1x1(res2)
         
        """
        if config.network.fpn_with_gap:
            fpn_gap = self.fpn_gap(F.adaptive_avg_pool2d(res5, (1, 1)).squeeze()).view(-1, self.feature_dim, 1, 1)
            fpn_p5_1x1 = fpn_p5_1x1 + fpn_gap
        """
        fpn_p5_upsample = self.fpn_upsample(fpn_p5_1x1)
        fpn_p4_plus = fpn_p5_upsample + fpn_p4_1x1
        fpn_p4_upsample = self.fpn_upsample(fpn_p4_plus)
        fpn_p3_plus = fpn_p4_upsample + fpn_p3_1x1
        fpn_p3_upsample = self.fpn_upsample(fpn_p3_plus)
        fpn_p2_plus = fpn_p3_upsample + fpn_p2_1x1

        fpn_p5 = self.fpn_p5(fpn_p5_1x1)
        fpn_p4 = self.fpn_p4(fpn_p4_plus)
        fpn_p3 = self.fpn_p3(fpn_p3_plus)
        fpn_p2 = self.fpn_p2(fpn_p2_plus)

        if hasattr(self, 'fpn_p6'):
            fpn_p6 = self.fpn_p6(fpn_p5)
            return fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6
        else:
            return fpn_p2, fpn_p3, fpn_p4, fpn_p5
    
class FPN_noup(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 with_extra_level=False,
                 with_norm='none', 
                 upsample_method='nearest'):
        super(FPN_noup, self).__init__()
        self.feature_dim = feature_dim
        assert upsample_method in ['nearest', 'bilinear']

        def interpolate(input):
            return F.interpolate(input, 
                                 scale_factor=2, 
                                 mode=upsample_method, 
                                 align_corners=False if upsample_method == 'bilinear' else None)
        self.fpn_upsample = interpolate

        assert with_norm in ['group_norm', 'batch_norm', 'none']
        if with_extra_level:
            self.fpn_p6 = nn.MaxPool2d(kernel_size=1, stride=2)
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm

        if with_norm != 'none':
            self.fpn_p5_1x1 = nn.Sequential(*[nn.Conv2d(2048, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p4_1x1 = nn.Sequential(*[nn.Conv2d(1024, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p3_1x1 = nn.Sequential(*[nn.Conv2d(512, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p2_1x1 = nn.Sequential(*[nn.Conv2d(256, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p5 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p4 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p3 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p2 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
        else:
            self.fpn_p5_1x1 = nn.Conv2d(2048, feature_dim, 1)
            self.fpn_p4_1x1 = nn.Conv2d(1024, feature_dim, 1)
            self.fpn_p3_1x1 = nn.Conv2d(512, feature_dim, 1)
            self.fpn_p2_1x1 = nn.Conv2d(256, feature_dim, 1)
            self.fpn_p5 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p4 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p3 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p2 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, res2, res3, res4, res5):
        fpn_p5_1x1 = self.fpn_p5_1x1(res5)
        fpn_p4_1x1 = self.fpn_p4_1x1(res4)
        fpn_p3_1x1 = self.fpn_p3_1x1(res3)
        if res3.shape[2]*2 != res2.shape[3]:
            res2 = F.interpolate(res2, size=(res3.shape[-2]*2, res3.shape[-1]*2), mode='bilinear', align_corners=False) 
        fpn_p2_1x1 = self.fpn_p2_1x1(res2)
      
    
        fpn_p4_plus = fpn_p5_1x1 + fpn_p4_1x1
        fpn_p3_plus = fpn_p4_plus + fpn_p3_1x1
        fpn_p3_upsample = self.fpn_upsample(fpn_p3_plus)
        fpn_p2_plus = fpn_p3_upsample + fpn_p2_1x1
        
        fpn_p5 = self.fpn_p5(fpn_p5_1x1)
        fpn_p4 = self.fpn_p4(fpn_p4_plus)
        fpn_p3 = self.fpn_p3(fpn_p3_plus)
        fpn_p2 = self.fpn_p2(fpn_p2_plus)

        if hasattr(self, 'fpn_p6'):
            fpn_p6 = self.fpn_p6(fpn_p5)
            return fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6
        else:
            return fpn_p2, fpn_p3, fpn_p4, fpn_p5
        return feat
    
class ResizeFPN(nn.Module):

    def __init__(self, 
                 feature_dim, 
                 out_size,
                 with_extra_level=False, 
                 with_norm='batch_norm', 
                 upsample_method='nearest'):
        super(ResizeFPN, self).__init__()
        self.feature_dim = feature_dim
        self.out_size = out_size
        assert upsample_method in ['nearest', 'bilinear']

        assert with_norm in ['group_norm', 'batch_norm', 'none']
        if with_extra_level:
            self.fpn_p6 = nn.MaxPool2d(kernel_size=1, stride=2)
        #if config.network.fpn_with_gap:
        #    self.fpn_gap = nn.Linear(2048, feature_dim)
        
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm

            
        if with_norm != 'none':
            self.fpn_p5_1x1 = nn.Sequential(*[nn.Conv2d(2048, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p4_1x1 = nn.Sequential(*[nn.Conv2d(1024, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p3_1x1 = nn.Sequential(*[nn.Conv2d(512, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p2_1x1 = nn.Sequential(*[nn.Conv2d(256, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p5 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p4 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p3 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpn_p2 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            #self.fpn_p2 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            
        else:
            self.fpn_p5_1x1 = nn.Conv2d(2048, feature_dim, 1)
            self.fpn_p4_1x1 = nn.Conv2d(1024, feature_dim, 1)
            self.fpn_p3_1x1 = nn.Conv2d(512, feature_dim, 1)
            self.fpn_p2_1x1 = nn.Conv2d(256, feature_dim, 1)
            self.fpn_p5 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p4 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p3 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
            self.fpn_p2 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, res2, res3, res4, res5):
        fpn_p5_1x1 = self.fpn_p5_1x1(res5)
        fpn_p5_1x1 = F.interpolate(fpn_p5_1x1, 
                                   size=self.out_size, 
                                   mode='bilinear', 
                                   align_corners=False) 
        fpn_p4_1x1 = self.fpn_p4_1x1(res4)
        fpn_p4_1x1 = F.interpolate(fpn_p4_1x1, 
                                   size=self.out_size, 
                                   mode='bilinear', 
                                   align_corners=False) 
        fpn_p3_1x1 = self.fpn_p3_1x1(res3)
        fpn_p3_1x1 = F.interpolate(fpn_p3_1x1, 
                                   size=self.out_size, 
                                   mode='bilinear', 
                                   align_corners=False) 
        fpn_p2_1x1 = self.fpn_p2_1x1(res2)
        fpn_p2_1x1 = F.interpolate(fpn_p2_1x1, 
                                   size=self.out_size, 
                                   mode='bilinear', 
                                   align_corners=False) 
         
        """
        if config.network.fpn_with_gap:
            fpn_gap = self.fpn_gap(F.adaptive_avg_pool2d(res5, (1, 1)).squeeze()).view(-1, self.feature_dim, 1, 1)
            fpn_p5_1x1 = fpn_p5_1x1 + fpn_gap
        """
        fpn_p4_plus = fpn_p5_1x1 + fpn_p4_1x1
        fpn_p3_plus = fpn_p4_plus + fpn_p3_1x1
        fpn_p2_plus = fpn_p3_plus + fpn_p2_1x1

        fpn_p5 = self.fpn_p5(fpn_p5_1x1)
        fpn_p4 = self.fpn_p4(fpn_p4_plus)
        fpn_p3 = self.fpn_p3(fpn_p3_plus)
        fpn_p2 = self.fpn_p2(fpn_p2_plus)

        if hasattr(self, 'fpn_p6'):
            fpn_p6 = self.fpn_p6(fpn_p5)
            return fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6
        else:
            return fpn_p2, fpn_p3, fpn_p4, fpn_p5
class mobileFPN_2(nn.Module):
    def __init__(self, 
                 feature_dim,
                 out_size,
                 input_dim=[320, 24],
                 with_norm='batch_norm'):
        super(mobileFPN_2, self).__init__()
        self.feature_dim = feature_dim
        self.out_size = out_size
        self.fpn_feature_num = 2
        assert len(input_dim) == 2
        assert with_norm in ['group_norm', 'batch_norm', 'none']
        
        if with_norm == 'batch_norm':
            self.norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            self.norm = group_norm

        #if with_norm != 'none':
        self.fpn_1_1x1 = nn.Sequential(*[nn.Conv2d(input_dim[0], feature_dim, 1, bias=False), self.norm(feature_dim)])
        self.fpn_2_1x1 = nn.Sequential(*[nn.Conv2d(input_dim[1], feature_dim, 1, bias=False), self.norm(feature_dim)])
        
        self.fpn_1 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), self.norm(feature_dim)])
        self.fpn_2 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), self.norm(feature_dim)])
            
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, mobile_high, mobile_low):
        fpn_1_1x1 = self.fpn_1_1x1(mobile_high)
        fpn_1_1x1 = F.interpolate(fpn_1_1x1, 
                                  size=self.out_size, 
                                  mode='bilinear', 
                                  align_corners=False) 
        fpn_2_1x1 = self.fpn_2_1x1(mobile_low)
        fpn_2_1x1 = F.interpolate(fpn_2_1x1, 
                                  size=self.out_size, 
                                  mode='bilinear', 
                                  align_corners=False) 
        
        fpn_2_plus = fpn_1_1x1 + fpn_2_1x1
        
        fpn_1 = self.fpn_1(fpn_1_1x1)
        fpn_2 = self.fpn_2(fpn_2_plus)   
        return fpn_1, fpn_2
    
class mobileFPN_3(mobileFPN_2):
    def __init__(self, 
                 feature_dim,
                 out_size,
                 input_dim=[320, 96, 24],
                 with_norm='batch_norm'):
        super().__init__(feature_dim, out_size, [input_dim[0], input_dim[1]], with_norm)
        self.fpn_feature_num = 3
        
        self.fpn_3_1x1 = nn.Sequential(*[nn.Conv2d(input_dim[2], feature_dim, 1, bias=False), self.norm(feature_dim)])
        self.fpn_3 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), self.norm(feature_dim)])
        self.initialize()
        
    def forward(self, mobile_1, mobile_2, mobile_3):
        fpn_1_1x1 = self.fpn_1_1x1(mobile_1)
        fpn_1_1x1 = F.interpolate(fpn_1_1x1, 
                                  size=self.out_size, 
                                  mode='bilinear', 
                                  align_corners=False) 
        fpn_2_1x1 = self.fpn_2_1x1(mobile_2)
        fpn_2_1x1 = F.interpolate(fpn_2_1x1, 
                                  size=self.out_size, 
                                  mode='bilinear', 
                                  align_corners=False)
        fpn_3_1x1 = self.fpn_3_1x1(mobile_3)
        fpn_3_1x1 = F.interpolate(fpn_3_1x1, 
                                  size=self.out_size, 
                                  mode='bilinear', 
                                  align_corners=False)
        
        fpn_2_plus = fpn_1_1x1 + fpn_2_1x1
        fpn_3_plus = fpn_2_plus + fpn_3_1x1
        fpn_1 = self.fpn_1(fpn_1_1x1)
        fpn_2 = self.fpn_2(fpn_2_plus)
        fpn_3 = self.fpn_3(fpn_3_plus)
        return fpn_1, fpn_2, fpn_3
        
class mobileFPN_4(mobileFPN_3):
    def __init__(self, 
                 feature_dim,
                 out_size=[17, 17],
                 input_dim=[320, 96, 32, 16],
                 with_norm='batch_norm'):
        super().__init__(feature_dim, out_size, [input_dim[0], input_dim[1], input_dim[2]], with_norm)
        self.fpn_feature_num = 4
        
        self.fpn_4_1x1 = nn.Sequential(*[nn.Conv2d(input_dim[3], feature_dim, 1, bias=False), self.norm(feature_dim)])
        self.fpn_4 = nn.Sequential(*[nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False), self.norm(feature_dim)])
        self.initialize()
        
    def forward(self, mobile_1, mobile_2, mobile_3, mobile_4):
        fpn_1_1x1 = self.fpn_1_1x1(mobile_1)
        fpn_1_1x1 = F.interpolate(fpn_1_1x1, 
                                  size=self.out_size, 
                                  mode='bilinear', 
                                  align_corners=False) 
        fpn_2_1x1 = self.fpn_2_1x1(mobile_2)
        fpn_2_1x1 = F.interpolate(fpn_2_1x1, 
                                  size=self.out_size, 
                                  mode='bilinear', 
                                  align_corners=False)
        fpn_3_1x1 = self.fpn_3_1x1(mobile_3)
        fpn_3_1x1 = F.interpolate(fpn_3_1x1, 
                                  size=self.out_size, 
                                  mode='bilinear', 
                                  align_corners=False)
        fpn_4_1x1 = self.fpn_4_1x1(mobile_4)
        fpn_4_1x1 = F.interpolate(fpn_4_1x1, 
                                  size=self.out_size, 
                                  mode='bilinear', 
                                  align_corners=False)
        
        fpn_2_plus = fpn_1_1x1 + fpn_2_1x1
        fpn_3_plus = fpn_2_plus + fpn_3_1x1
        fpn_4_plus = fpn_3_plus + fpn_4_1x1
        fpn_1 = self.fpn_1(fpn_1_1x1)
        fpn_2 = self.fpn_2(fpn_2_plus)
        fpn_3 = self.fpn_3(fpn_3_plus)
        fpn_4 = self.fpn_4(fpn_4_plus)
        return fpn_1, fpn_2, fpn_3, fpn_4            
           
