# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import sys
sys.path.append('modules')
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from ASPP import ASPP
import xception as xception

class deeplabv3plus(nn.Module):
    def __init__(self, pretrained=True, num_classes=-1):
        super(deeplabv3plus, self).__init__()
        self.backbone = None        
        self.backbone_layers = None
        input_channel = 2048        
        self.aspp = ASPP(dim_in=input_channel, 
                dim_out=256, 
                rate=16//16,
                bn_mom = 0.0003)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=16//8) #16//4

        indim = 728
        shallow1_dim = 64
        self.shortcut_conv1_1 = nn.Sequential(
                nn.Conv2d(indim, shallow1_dim, 1, 1, padding=1//2,bias=True),
                SynchronizedBatchNorm2d(shallow1_dim, momentum=0.0003),
                nn.ReLU(inplace=True),      
        )       
        self.cat_conv1_1 = nn.Sequential(
                nn.Conv2d(256+shallow1_dim, 256, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(256, momentum=0.0003),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(256, 256, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(256, momentum=0.0003),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        )
        indim = 256
        shallow2_dim = 32
        self.shortcut_conv1_2 = nn.Sequential(
                nn.Conv2d(indim, shallow2_dim, 1, 1, padding=1//2,bias=True),
                SynchronizedBatchNorm2d(shallow2_dim, momentum=0.0003),
                nn.ReLU(inplace=True),
        )
        self.cat_conv1_2 = nn.Sequential(
                nn.Conv2d(256+shallow2_dim, 256, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(256, momentum=0.0003),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(256, 256, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(256, momentum=0.0003),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        )
        # self.predict5x5 = nn.Conv2d(256, 256, 5, 1, padding=2)
        self.predict5x5 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(256, momentum=0.0003),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(256, 256, 3, 1, padding=1,bias=True),
                SynchronizedBatchNorm2d(256, momentum=0.0003),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
        )

        self.cls_conv = nn.Conv2d(256, num_classes, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = xception.xception(pretrained = pretrained, os=16)
        self.backbone_layers = self.backbone.get_layers()

    def forward(self, x):
        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
#        print('ASPP:',feature_aspp.shape)
        #1/8
        feature_aspp = self.upsample_sub(feature_aspp)
        feature_shallow = self.shortcut_conv1_1(layers[1])
        feature_cat = torch.cat([feature_aspp,feature_shallow],1)
        result = self.cat_conv1_1(feature_cat) 
#        print('upsample1:', result.shape)
        #1/4
        feature_aspp = self.upsample_sub(result)
        feature_shallow = self.shortcut_conv1_2(layers[0])
        feature_cat = torch.cat([feature_aspp,feature_shallow],1)
        result = self.cat_conv1_2(feature_cat)
#        print('upsample2:', result.shape)

        result = self.predict5x5(result)
        result = self.cls_conv(result)
        result = self.upsample4(result)
#        print('final:', result.shape)
        return result, result

