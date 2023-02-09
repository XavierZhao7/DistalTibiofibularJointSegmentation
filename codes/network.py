import pathlib
import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import random
from skimage import io
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import shutil
from sklearn.metrics import accuracy_score, recall_score, auc, roc_curve, precision_score
import warnings
warnings.filterwarnings('ignore')



# With nn.Upsample, nn.ReLU, 4 downsample and the bottleneck features = 1024
# More original one 
class UNet_v1(nn.Module):

    def __init__(self, nchannel=3, nclass=3):
        super().__init__()
        self.iput = self.conv(nchannel, 64)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.enc1 = self.conv(64, 128)
        self.enc2 = self.conv(128, 256)
        self.enc3 = self.conv(256, 512)
        self.enc4 = self.conv(512, 1024 // 2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv(1024, 512 // 2)
        self.dec2 = self.conv(512, 256 // 2)
        self.dec3 = self.conv(256, 128 // 2)
        self.dec4 = self.conv(128, 64)
        self.oput = torch.nn.Conv2d(64, nclass, kernel_size=1)

    def forward(self, x):
        x1 = self.iput(x)  # input
        # encoder layers
        x2 = self.maxpool(x1)
        x2 = self.enc1(x2)
        x3 = self.maxpool(x2)
        x3 = self.enc2(x3)
        x4 = self.maxpool(x3)
        x4 = self.enc3(x4)
        x5 = self.maxpool(x4)
        x5 = self.enc4(x5)
        # decoder layers with skip connections and attention gates
        x = self.upsample(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec1(x)
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec4(x)
        return self.oput(x)  # output

    def conv(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True))



class Spatial_Attention_define(nn.Module):
    def __init__(self, num_channels):
        super(Spatial_Attention_define, self).__init__()

        self.conv = nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(num_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_conv1 = self.conv1(x)
        x = torch.cat([avg_out, max_out, x_conv1], dim=1)
        psi = self.conv(x)

        return psi

class Spatial_Attention_gate(nn.Module):
    def __init__(self, F_g, F_l):
        super(Spatial_Attention_gate, self).__init__()

        self.W_g = Spatial_Attention_define(num_channels=F_g)

        self.W_x = Spatial_Attention_define(num_channels=F_l)

        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.sigmoid(g1 + x1)

        return psi
        
class AttU_Net_sAG(nn.Module):
    def __init__(self, nchannel=3, nclass=3):
        super(AttU_Net_sAG, self).__init__()
        self.iput = self.conv(nchannel, 64)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.enc1 = self.conv(64, 128)
        self.enc2 = self.conv(128, 256)
        self.enc3 = self.conv(256, 512)
        self.enc4 = self.conv(512, 1024 // 2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv(1024, 512 // 2)
        self.dec2 = self.conv(512, 256 // 2)
        self.dec3 = self.conv(256, 128 // 2)
        self.dec4 = self.conv(128, 64)
        self.oput = torch.nn.Conv2d(64, nclass, kernel_size=1)

        self.Spa5 = Spatial_Attention_gate(512, 512)
        self.Spa4 = Spatial_Attention_gate(256, 256)
        self.Spa3 = Spatial_Attention_gate(128, 128)
        self.Spa2 = Spatial_Attention_gate(64, 64)

    def forward(self, x):
        x1 = self.iput(x)  # input
        # encoder layers
        x2 = self.maxpool(x1)
        x2 = self.enc1(x2)
        x3 = self.maxpool(x2)
        x3 = self.enc2(x3)
        x4 = self.maxpool(x3)
        x4 = self.enc3(x4)
        x5 = self.maxpool(x4)
        x5 = self.enc4(x5)
        # decoder layers with skip connections and attention gates
        x = self.upsample(x5)
        s4 = self.Spa5(x, x4)
        x4 = x4 * s4
        x = torch.cat([x, x4], dim=1)

        x = self.dec1(x)
        x = self.upsample(x)
        s3 = self.Spa4(x, x3)
        x3 = x3 * s3
        x = torch.cat([x, x3], dim=1)

        x = self.dec2(x)
        x = self.upsample(x)
        s2 = self.Spa3(x, x2)
        x2 = x2 * s2
        x = torch.cat([x, x2], dim=1)

        x = self.dec3(x)
        x = self.upsample(x)
        s1 = self.Spa2(x, x1)
        x1 = x1 * s1
        x = torch.cat([x, x1], dim=1)

        x = self.dec4(x)
        return self.oput(x), s1  # output and attention map

    def conv(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True))
        


# With nn.Upsample, nn.ReLU, 4 downsample and the bottleneck features = 1024
# More original one 
class UNet3d_v1(nn.Module):

    def __init__(self, nchannel=3, nclass=3):
        super().__init__()
        self.iput = self.conv(nchannel, 64)
        self.maxpool = torch.nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1))
        self.enc1 = self.conv(64, 128)
        self.enc2 = self.conv(128, 256)
        self.enc3 = self.conv(256, 512)
        self.enc4 = self.conv(512, 1024 // 2)

        self.upsample1 = nn.ConvTranspose3d(512, 512, 
                                           kernel_size=(2,2,1), stride=(2, 2, 1),
                                           padding=(0,0,0))
        self.upsample2 = nn.ConvTranspose3d(256, 256, 
                                    kernel_size=(2,2,1), stride=(2, 2, 1),
                                    padding=(0,0,0))
        self.upsample3 = nn.ConvTranspose3d(128, 128, 
                            kernel_size=(2,2,1), stride=(2, 2, 1),
                            padding=(0,0,0))
        self.upsample4 = nn.ConvTranspose3d(64, 64, 
                    kernel_size=(2,2,1), stride=(2, 2, 1),
                    padding=(0,0,0))
        
        # self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv(1024, 512 // 2)
        self.dec2 = self.conv(512, 256 // 2)
        self.dec3 = self.conv(256, 128 // 2)
        self.dec4 = self.conv(128, 64)

        self.oput = torch.nn.Conv3d(64, nclass, kernel_size=1)

    def forward(self, x):
        x1 = self.iput(x)  # input
        # encoder layers
        x2 = self.maxpool(x1)
        x2 = self.enc1(x2)
        x3 = self.maxpool(x2)
        x3 = self.enc2(x3)
        x4 = self.maxpool(x3)
        x4 = self.enc3(x4)
        x5 = self.maxpool(x4)
        x5 = self.enc4(x5)
        # decoder layers with skip connections and attention gates
        x = self.upsample1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec1(x)
        x = self.upsample2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)
        x = self.upsample3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)
        x = self.upsample4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec4(x)
        return self.oput(x)  # output

    def conv(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(inplace=True))
