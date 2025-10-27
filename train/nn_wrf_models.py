#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:53:31 2024

@author: manuel

Model with U-Net achitecture following Ronneberger et al (2015) with four encoder blocks

"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn.functional import relu, leaky_relu
from torch.nn import functional as F

#from torchvision import models

import os




class unet_devine_circpad_lrelu_4pool(nn.Module):
    
    def __init__(self, layers_in, layers_out):
        super().__init__()
        
        
        feat_start = 32        # number of feature maps in first encoder blocks (doubled in each subsequent block)
        
        # define layers used in unet 
        # Encoder block 1
        self.e11 = nn.Conv2d(layers_in, feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.e12 = nn.Conv2d(feat_start, feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder block 2
        self.e21 = nn.Conv2d(feat_start, 2*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.e22 = nn.Conv2d(2*feat_start, 2*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder block 3
        self.e31 = nn.Conv2d(2*feat_start, 4*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.e32 = nn.Conv2d(4*feat_start, 4*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        # Encoder block 4
        self.e41 = nn.Conv2d(4*feat_start, 8*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.e42 = nn.Conv2d(8*feat_start, 8*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottle neck
        self.e51 = nn.Conv2d(8*feat_start, 16*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.e52 = nn.Conv2d(16*feat_start, 16*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        
        # Decoder block 1
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.d11 = nn.Conv2d(16*feat_start, 8*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.d12 = nn.Conv2d(16*feat_start, 8*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.d13 = nn.Conv2d(8*feat_start, 8*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        
        # Decoder block 2
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.d21 = nn.Conv2d(8*feat_start, 4*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.d22 = nn.Conv2d(8*feat_start, 4*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.d23 = nn.Conv2d(4*feat_start, 4*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        
        # Decoder block 3
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.d31 = nn.Conv2d(4*feat_start, 2*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.d32 = nn.Conv2d(4*feat_start, 2*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.d33 = nn.Conv2d(2*feat_start, 2*feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        
        # Decoder block 4
        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.d41 = nn.Conv2d(2*feat_start, feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.d42 = nn.Conv2d(2*feat_start, feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.d43 = nn.Conv2d(feat_start, feat_start, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        
        # output convolution
        self.outconv = nn.Conv2d(feat_start, layers_out, kernel_size=3, stride=1, padding=1, padding_mode='circular')
    
    def forward(self, x):
        
        # set up architecture
        ## ENCODER
        xe11 = leaky_relu(self.e11(x))
        xe12 = leaky_relu(self.e12(xe11))
        xp1 = self.pool1(xe12)
        
        xe21 = leaky_relu(self.e21(xp1))
        xe22 = leaky_relu(self.e22(xe21))
        xp2 = self.pool2(xe22)
        
        xe31 = leaky_relu(self.e31(xp2))
        xe32 = leaky_relu(self.e32(xe31))
        xp3 = self.pool3(xe32)
        
        xe41 = leaky_relu(self.e41(xp3))
        xe42 = leaky_relu(self.e42(xe41))
        xp4 = self.pool4(xe42)
        
        xe51 = leaky_relu(self.e51(xp4))
        xe52 = leaky_relu(self.e52(xe51))
        
        ## DECODER
        xup1 = self.up1(xe52)
        xd11 = leaky_relu(self.d11(xup1))
        xc11 = torch.cat([xd11, xe42], dim=1) # skipconnection
        xd12 = leaky_relu(self.d12(xc11))
        xd13 = leaky_relu(self.d13(xd12))
        
        xup2 = self.up2(xd13)
        xd21 = leaky_relu(self.d21(xup2))
        xc21 = torch.cat([xd21, xe32], dim=1) # skipconnection
        xd22 = leaky_relu(self.d22(xc21))
        xd23 = leaky_relu(self.d23(xd22))
        
        xup3 = self.up3(xd23)
        xd31 = leaky_relu(self.d31(xup3))
        xc31 = torch.cat([xd31, xe22], dim=1) # skipconnection
        xd32 = leaky_relu(self.d32(xc31))
        xd33 = leaky_relu(self.d33(xd32))
        
        xup4 = self.up4(xd33)
        xd41 = leaky_relu(self.d41(xup4))
        xc41 = torch.cat([xd41, xe12], dim=1) # skipconnection
        xd42 = leaky_relu(self.d42(xc41))
        xd43 = leaky_relu(self.d43(xd42))
        
        out = self.outconv(xd43)
        
        return out
    

