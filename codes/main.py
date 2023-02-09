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
from dataset import CBCTdataset
from train import Train, Train_3DUNet, Train_3DUNet_2D, Train_att
from network import UNet_v1, UNet3d_v1, AttU_Net_sAG
from evaluate import evaluate, evaluate_3d, evaluate_att


drive = pathlib.Path('./drive/MyDrive') / 'DLMI_Final/' ### CHANGE YOUR MAIN FILE PATH
database = drive/'Data/data_3'   # BEFORE TRAINING, CHANGE DATASET PATH TO CORRECT ONE


# dataloader and initialization randomness
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(8)
torch.manual_seed(8);


# train
# customize your own parameters
Train_3DUNet_2D(UNet3d_v1, 'result/UNet_3dunet2d.pth', bz = 5,lr = 0.01)
Train(UNet_v1, 'result/UNet_8.pth', bz=5, lr=0.01)
Train_att(AttU_Net_sAG,'result/Att_UNet1.pth', bz=5, lr=0.01)



# evaluate

evaluate_3d(UNet3d_v1, 'result/UNet_3dunet2d.pth',dataset='test')
evaluate(UNet_v1, 'result/UNet_8.pth',dataset='test')
evaluate_att(AttU_Net_sAG, 'result/Att_UNet1.pth',dataset='test')




