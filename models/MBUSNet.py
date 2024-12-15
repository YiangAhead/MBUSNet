import torch.utils.data
import timm
import sys
from collections import OrderedDict
import torch
import functools
import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F
import time
from torch.autograd import Variable
import torchvision.models as models
from MDAF import MDAF


class Dual_MDAF_uaca_BD_init(nn.Module):
    def __init__(self):
        super(Dual_MDAF_uaca_BD_init, self).__init__()

        self.fusions = [not True, not True, True, True, True]

        self.inplanes = [64, 128, 256, 512]

        rgb_backbone = models.resnet34(pretrained=True)
        rgb_backbone.load_state_dict(torch.load('/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/yinhanlong/MBUS_segmentation/pretrained_model/resnet34-b627a593.pth'))
        th_backbone = models.resnet34(pretrained=True)
        th_backbone.load_state_dict(torch.load('/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/yinhanlong/MBUS_segmentation/pretrained_model/resnet34-b627a593.pth'))

        # self.rgb_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.rgb_conv1.weight.data = torch.mean(rgb_backbone.conv1.weight.data, dim=1,keepdim=True)
 
        self.rgb_layer0 = nn.Sequential(
            rgb_backbone.conv1,
            rgb_backbone.bn1,
            rgb_backbone.relu,
            rgb_backbone.maxpool,
        )

        self.rgb_layer1 = rgb_backbone.layer1
        self.rgb_layer2 = rgb_backbone.layer2
        self.rgb_layer3 = rgb_backbone.layer3
        self.rgb_layer4 = rgb_backbone.layer4

        # self.th_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.th_conv1.weight.data = torch.mean(th_backbone.conv1.weight.data, dim=1,keepdim=True)
        self.th_layer0 = nn.Sequential(
            th_backbone.conv1,
            th_backbone.bn1,
            th_backbone.relu,
            th_backbone.maxpool,
        )
        self.th_layer1 = th_backbone.layer1
        self.th_layer2 = th_backbone.layer2
        self.th_layer3 = th_backbone.layer3
        self.th_layer4 = th_backbone.layer4

        ### Decoder 
        self.up32to16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.inplanes[-1], self.inplanes[-2], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes[-2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.inplanes[-2], self.inplanes[-3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes[-3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.inplanes[-3], self.inplanes[-4], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes[-4]),
            nn.ReLU(inplace=True),
        )
        self.up16to8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.inplanes[-2], self.inplanes[-3], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes[-3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.inplanes[-3], self.inplanes[-4], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes[-4]),
            nn.ReLU(inplace=True),
        )
        self.up8to4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.inplanes[-3], self.inplanes[-4], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes[-4]),
            nn.ReLU(inplace=True),
        )
        
        self.up4to4 = nn.Sequential(
            nn.Conv2d(self.inplanes[-4], self.inplanes[-4], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes[-4]),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(self.inplanes[-4], self.inplanes[-4], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes[-4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes[-4], 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.efusion2 = MDAF_EFusion(self.inplanes[-4])
        self.efusion4 = MDAF_EFusion(self.inplanes[-4])
        self.efusion8 = MDAF_EFusion(self.inplanes[-3])
        self.efusion16 = MDAF_EFusion(self.inplanes[-2])
        self.efusion32 = MDAF_EFusion(self.inplanes[-1])
        
        self.feat4_conv = nn.Sequential(
            nn.Conv2d(4 * self.inplanes[-4], self.inplanes[-4], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.inplanes[-4]),
            nn.ReLU(inplace=True),
        )
        
        self.uaca= UACA(64,32)   
        
        self.mask = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)
        
        #增加边界深监督
        self.predict1 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)
        self.predict2 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)
        self.predict3 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)
        self.predict4 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)
        
        self.maxpool = nn.MaxPool2d(3,stride=1,padding=1)
    def forward(self, rgb, thermal):
        rgb_feat2 = self.rgb_layer0(rgb)
        th_feat2 = self.th_layer0(thermal)

        rgb_feat4 = self.rgb_layer1(rgb_feat2)
        th_feat4 = self.th_layer1(th_feat2)
        feat4 = self.efusion4(rgb_feat4, th_feat4, fusion=self.fusions[1]) 

        rgb_feat8 = self.rgb_layer2(rgb_feat4)
        th_feat8 = self.th_layer2(th_feat4)
        feat8 = self.efusion8(rgb_feat8, th_feat8, fusion=self.fusions[2])
      
        rgb_feat16 = self.rgb_layer3(rgb_feat8)
        th_feat16 = self.th_layer3(th_feat8)
        feat16 = self.efusion16(rgb_feat16, th_feat16, fusion=self.fusions[3])

        rgb_feat32 = self.rgb_layer4(rgb_feat16)
        th_feat32 = self.th_layer4(th_feat16)
        feat32 = self.efusion32(rgb_feat32, th_feat32, fusion=self.fusions[4])
        
        feat32_up = self.up32to16(feat32)
        feat16_up = self.up16to8(feat16)
        feat8_up = self.up8to4(feat8)
        feat4_up = self.up4to4(feat4)
        
        feat32_up = self.up32to16(feat32)
        feat16_up = self.up16to8(feat16)
        feat8_up = self.up8to4(feat8)
        feat4_up = self.up4to4(feat4)
        
        b1 = self.predict1(feat4_up)
        b2 = self.predict2(feat8_up)
        b3 = self.predict3(feat16_up)
        b4 = self.predict4(feat32_up)
        
        
        b1 = F.interpolate(b1, size=(256,256), mode='bilinear', align_corners=False)
        b2 = F.interpolate(b2, size=(256,256), mode='bilinear', align_corners=False)
        b3 = F.interpolate(b3, size=(256,256), mode='bilinear', align_corners=False)
        b4 = F.interpolate(b4, size=(256,256), mode='bilinear', align_corners=False)
        
        p1_edge=b1
        smoothed_1=self.maxpool(p1_edge)
        edge1=torch.abs(p1_edge-smoothed_1)
        e_1=b1+edge1
        
        p2_edge = b2
        smoothed_2 = self.maxpool(p2_edge)
        edge2 = torch.abs(p2_edge - smoothed_2)
        e_2 = b2 + edge2
        
        p3_edge = b3
        smoothed_3 = self.maxpool(p3_edge)
        edge3 = torch.abs(p3_edge - smoothed_3)
        e_3 = b3 + edge3
        
        p4_edge = b4
        smoothed_4 = self.maxpool(p4_edge)
        edge4 = torch.abs(p4_edge - smoothed_4)
        e_4 = b4 + edge4

        feat4_up = torch.concat([feat32_up,feat16_up,feat8_up,feat4],dim=1)
        feat4_up = self.feat4_conv(feat4_up)
        

        init_out = self.head(feat4_up)
        # out=F.interpolate(out,size=[rgb.shape[-2], rgb.shape[-1]],mode='bilinear')
        
        _,out = self.uaca(feat4_up,init_out)
        
        out = F.interpolate(out, size=(256,256), mode='bilinear', align_corners=False)
        init_out = F.interpolate(init_out, size=(256,256), mode='bilinear', align_corners=False)
  
        return out.sigmoid(),e_1,e_2,e_3,e_4,init_out
