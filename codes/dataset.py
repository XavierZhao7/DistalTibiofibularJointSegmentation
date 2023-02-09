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








class CBCTdataset(Dataset):
    def __init__(self,
                 input_dir,
                 mode,
                 transform = None):

        self.mode = mode
        self.transform = transform
        try:
            if self.mode == 'train':
                self.data_dir = os.path.join(input_dir, 'train')
            elif self.mode == 'val':
                self.data_dir = os.path.join(input_dir, 'valid')
            elif self.mode == 'test':
                self.data_dir = os.path.join(input_dir, 'test')
        except ValueError:
            print('op should be either train, val or test!')

    def __len__(self):

        return len(os.listdir(f"{self.data_dir}/image"))

    def __getitem__(self,
                    idx):
        
        self.image_list, self.mask_list = self.get_data_name_list(self.data_dir)
        # self.image_name = self.image_list[idx].split('/')[-1]
        # self.mask_name = self.mask_list[idx].split('/')[-1]
        # print(self.image_list)

        # img = io.imread(self.image_list[idx])
        img = np.load(self.image_list[idx],allow_pickle=1)

        mask = np.load(self.mask_list[idx],allow_pickle=1)

        ## Transform image and mask
        if self.transform:
            img = self.normalize(img)
            if img.shape[0] == 5:
                img, mask = self.img_transform3d(img, mask)
            else:
                img, mask = self.img_transform(img, mask)
            # img = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))(img)
  
        return img, mask

    def img_transform(self,
                      img,
                      mask):

        ## Apply Transformations to Image and Mask
        # 3 channel image for combine
        new_image = np.zeros((mask.shape[0],mask.shape[1], mask.shape[2]))
        for c in range(3):
            new_image[:,:,c] = img

        img = new_image
        # one-hot coding the mask for the 3 classes
        for c in range(3):
            mask[:,:,c] = mask[:,:,c] != 0

        img = img.astype(float).transpose(2,0,1)
        mask = mask.astype(float).transpose(2,0,1)
        
        img = torch.as_tensor(img.copy()).unsqueeze(0) 
        mask = torch.as_tensor(mask.copy()).unsqueeze(0)
        
        combined = torch.cat([img, mask], dim=0)


        combined = self.transform(combined)

        # Input is 3 channel or 1? now 3
        image2 = combined[0,0:3,:,:]

        mask2 = combined[1,0:3,:,:]
        img, mask = image2, mask2

        return img, mask

    def img_transform3d(self,
                      img,
                      mask):
        
        new_image = np.zeros((mask.shape[0],mask.shape[1], mask.shape[2],\
                              mask.shape[3]))
        for c in range(3):
            new_image[:,:,:,c] = img

        img = new_image
        # one-hot coding the mask for the 3 classes
        for c in range(3):
            mask[:,:,:,c] = mask[:,:,:,c] != 0

        img = img.astype(float).transpose(3,1,2,0)
        mask = mask.astype(float).transpose(3,1,2,0)
        
        img = torch.as_tensor(img.copy()).unsqueeze(0) 
        mask = torch.as_tensor(mask.copy()).unsqueeze(0)
        
        combined = torch.cat([img, mask], dim=0)
        new_combined = np.zeros([2,3,96,96,5])
        for sl in range(5):
            new_combined[:,:,:,:,sl] = self.transform(combined[:,:,:,:,sl])

        combined = new_combined
        # Input is 3 channel or 1? now 3
        image2 = combined[0,0:3,:,:,:]

        mask2 = combined[1,0:3,:,:,:]
        img, mask = image2, mask2

        return img, mask

    def get_data_name_list(self, data_dir):

        image_path = os.path.join(data_dir, 'image')
        mask_path = os.path.join(data_dir, 'label')

        image_list = [os.path.join(image_path, image) for image in sorted(os.listdir(image_path))]
        mask_list = [os.path.join(mask_path, mask) for mask in sorted(os.listdir(mask_path))]

        return image_list, mask_list 

    def normalize(self, img):
        return (img-img.min())/(img.max()-img.min())