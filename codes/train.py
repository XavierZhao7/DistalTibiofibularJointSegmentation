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




warnings.filterwarnings('ignore')

def display_data(image,mask):
    plt.imshow(image[0,:,:,:].numpy().transpose(1,2,0))
    plt.figure()
    plt.imshow(mask[0,:,:,:].numpy().transpose(1,2,0))
    plt.show()



# square version
class BinearyDICELoss(nn.Module):
    def __init__(self):
        super(BinearyDICELoss, self).__init__()

    def forward(self, pred, target, smooth=1):
        # print(pred.shape)
        # print(pred.max(),pred.min())

        # print(pred.max(),pred.min())
        intersect = torch.mul(pred, target).sum(-1).sum(-1)
        denominator = (pred ** 2).sum(-1).sum(-1) + (target ** 2).sum(-1).sum(-1)
        dice_score = (2 * intersect + smooth) / (denominator + smooth)
        out = 1 - dice_score.mean()
        
        return out

class DICELoss(nn.Module):
    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self,pred, target):
        # pred = torch.sigmoid(pred)

        cl = target.shape[1]
        cost = BinearyDICELoss()
        loss = 0
        for i in range(cl):
            loss += cost(pred[:,i,:,:],target[:,i,:,:])
        loss /= cl
        return loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=0, 
                 verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        # if self.save_path:
        #   if not os.path.exists(self.save_path):
        #     os.mkdir(self.save_path)

    def __call__(self, save_path, epoch,loss_train,losses_val, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            if save_path:
              self.save_checkpoint(save_path, epoch, loss_train, losses_val, val_loss, model)

        elif score >= self.best_score + self.delta:
            self.counter += 1
            # if self.counter % 1 == 0:
            #     print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save_path:
              # print('++++++++++saving model...++++++++++')
              self.save_checkpoint(save_path, epoch, loss_train,losses_val, val_loss, model)
            self.counter = 0

    def save_checkpoint(self, save_path, epoch, loss_train, losses_val, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('=='*50)
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print('=='*50)

        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'losses_train': loss_train,
            'losses_val': losses_val}
        torch.save(checkpoint, drive / save_path)

        self.val_loss_min = val_loss
        # print(f'saveing model done... validation loss is: {self.val_loss_min:.6f}')


early_stopping = EarlyStopping(patience=100, verbose=False)

def Train(model, path, num_epochs=500, bz = 5, lr = 0.01, display=False):
    # transforms.RandomCrop for small patches
    # Same seed for each experiments 
    random.seed(8)
    torch.manual_seed(8)
    img_transform = transforms.Compose([transforms.Resize([96,96]),
        transforms.RandomVerticalFlip(0.5),transforms.RandomHorizontalFlip(0),
        transforms.RandomRotation(90)
    ])

    # hyperparameters
    device = torch.device('cuda')
    train_batch_size = bz
    validation_batch_size = bz
    n_classes = 3 # binary mask
    model = model().to(device)
    learning_rate = lr

    ## Initialize Dataloaders
    train_dataset=CBCTdataset(input_dir=database, mode="train",transform=img_transform)
    validation_dataset=CBCTdataset(input_dir=database, mode="val",transform=img_transform)
    test_dataset=CBCTdataset(input_dir=database, mode="test",transform=img_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    ## Initialize Optimizer and Learning Rate Scheduler
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # loss function 
    cost_bce = nn.BCEWithLogitsLoss()
    cost_dice = DICELoss()
    
    losses_train = []
    losses_val = []
    print("Start Training...")
    ## Start training 
    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss_train = 0
        loss_val = 0
        for image, mask in train_dataloader:
            image = image.float()
            mask = mask.float()
            if display:
              display_data(image,mask)
              break
            optimizer.zero_grad()
            predictions= model(image.to(device))

            # loss1 = cost_bce.forward(predictions,mask.to(device))
            loss2 = cost_dice.forward(predictions,mask.to(device))
            # loss = loss1 + loss2
            # print(loss1,loss2)

            loss2.backward()
            optimizer.step()
            loss_train += loss2.item()
        # scheduler.step()
        # loss for each batch 
        losses_train.append(loss_train/ len(train_dataloader))
        if (epoch+1) % 10 == 0:
            print(losses_train[-1])

    # validation loss
        with torch.no_grad():
          model.eval()
          predictions = []
          for image, mask in validation_dataloader:
              image = image.float()
              mask = mask.float()
              predictions = model(image.to(device))

              # loss1 = cost_bce.forward(predictions,mask.to(device))
              loss2 = cost_dice.forward(predictions,mask.to(device))
              # loss = loss1 + loss2
              loss_val += loss2.item()

          losses_val.append(loss_val/ len(validation_dataloader))

        early_stopping(path,epoch,losses_train,losses_val, loss_val/ len(validation_dataloader), model,
                       )
        if early_stopping.early_stop:
            print('++++++++++Early Stopping...++++++++++')
            break



early_stopping = EarlyStopping(patience=100, verbose=False)

def Train_att(model, path, num_epochs=500, bz = 10, lr = 0.01, display=False):
    # transforms.RandomCrop for small patches
    # Same seed for each experiments 
    random.seed(8)
    torch.manual_seed(8)
    img_transform = transforms.Compose([transforms.Resize([96,96]),
        transforms.RandomVerticalFlip(0.5),transforms.RandomHorizontalFlip(0),
        transforms.RandomRotation(90)
    ])

    # hyperparameters
    device = torch.device('cuda')
    train_batch_size = bz
    validation_batch_size = bz
    n_classes = 3 # binary mask
    model = model().to(device)
    learning_rate = lr

    ## Initialize Dataloaders
    train_dataset=CBCTdataset(input_dir=database, mode="train",transform=img_transform)
    validation_dataset=CBCTdataset(input_dir=database, mode="val",transform=img_transform)
    test_dataset=CBCTdataset(input_dir=database, mode="test",transform=img_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    ## Initialize Optimizer and Learning Rate Scheduler
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # loss function 
    cost = nn.BCEWithLogitsLoss()
    # cost = DICELoss()

    losses_train = []
    losses_val = []
    print("Start Training...")
    ## Start training 
    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss_train = 0
        loss_val = 0
        for image, mask in train_dataloader:
            image = image.float()
            mask = mask.float()
            if display:
              display_data(image,mask)
              break
            optimizer.zero_grad()
            predictions, att_map = model(image.to(device))

            loss = cost.forward(predictions,mask.to(device))
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        # scheduler.step()
        # loss for each batch 
        losses_train.append(loss_train/ len(train_dataloader))
        if (epoch+1) % 10 == 0:
            print(losses_train[-1])

    # validation loss
        with torch.no_grad():
          model.eval()
          predictions = []
          for image, mask in validation_dataloader:
              image = image.float()
              mask = mask.float()
              predictions,att_map = model(image.to(device))

              loss = cost.forward(predictions,mask.to(device))
              loss_val += loss.item()

          losses_val.append(loss_val/ len(validation_dataloader))

        early_stopping(path,epoch,losses_train,losses_val, loss_val/ len(validation_dataloader), model,
                       )
        if early_stopping.early_stop:
            print('++++++++++Early Stopping...++++++++++')
            break



early_stopping = EarlyStopping(patience=100, verbose=False)

def Train_3DUNet(model, path, num_epochs=500, bz = 5, lr = 0.01, display=False):
    # transforms.RandomCrop for small patches
    # Same seed for each experiments 
    random.seed(8)
    torch.manual_seed(8)
    img_transform = transforms.Compose([transforms.Resize([96,96]),
        transforms.RandomVerticalFlip(0.5),transforms.RandomHorizontalFlip(0),
        transforms.RandomRotation(90)
    ])

    # hyperparameters
    device = torch.device('cuda')
    train_batch_size = bz
    validation_batch_size = bz
    n_classes = 3 # binary mask
    model = model().to(device)
    learning_rate = lr

    ## Initialize Dataloaders
    train_dataset=CBCTdataset(input_dir=database, mode="train",transform=img_transform)
    validation_dataset=CBCTdataset(input_dir=database, mode="val",transform=img_transform)
    test_dataset=CBCTdataset(input_dir=database, mode="test",transform=img_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    ## Initialize Optimizer and Learning Rate Scheduler
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # loss function 
    cost_bce = nn.BCEWithLogitsLoss()
    cost_dice = DICELoss()
    
    losses_train = []
    losses_val = []
    print("Start Training...")
    ## Start training 
    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss_train = 0
        loss_val = 0
        for image, mask in train_dataloader:
            image = image.float()
            mask = mask.float()
            if display:
              display_data(image,mask)
              break
            optimizer.zero_grad()
            predictions= model(image.to(device))

            loss1 = cost_bce.forward(predictions,mask.to(device))
            # loss2 = cost_dice.forward(predictions,mask.to(device))
            # loss = loss1 + loss2
            # print(loss1,loss2)

            loss1.backward()
            optimizer.step()
            loss_train += loss1.item()
        # scheduler.step()
        # loss for each batch 
        losses_train.append(loss_train/ len(train_dataloader))
        if (epoch+1) % 10 == 0:
            print(losses_train[-1])

    # validation loss
        with torch.no_grad():
          model.eval()
          predictions = []
          for image, mask in validation_dataloader:
              image = image.float()
              mask = mask.float()
              predictions = model(image.to(device))

              loss1 = cost_bce.forward(predictions,mask.to(device))
              # loss2 = cost_dice.forward(predictions,mask.to(device))
              # loss = loss1 + loss2
              loss_val += loss1.item()

          losses_val.append(loss_val/ len(validation_dataloader))

        early_stopping(path,epoch,losses_train,losses_val, loss_val/ len(validation_dataloader), model,
                       )
        if early_stopping.early_stop:
            print('++++++++++Early Stopping...++++++++++')
            break


early_stopping = EarlyStopping(patience=100, verbose=False)

def Train_3DUNet_2D(model, path, num_epochs=500, bz = 5, lr = 0.01, display=False):
    # transforms.RandomCrop for small patches
    # Same seed for each experiments 
    random.seed(8)
    torch.manual_seed(8)
    img_transform = transforms.Compose([transforms.Resize([96,96]),
        transforms.RandomVerticalFlip(0.5),transforms.RandomHorizontalFlip(0),
        transforms.RandomRotation(90)
    ])

    # hyperparameters
    device = torch.device('cuda')
    train_batch_size = bz
    validation_batch_size = bz
    n_classes = 3 # binary mask
    model = model().to(device)
    learning_rate = lr

    ## Initialize Dataloaders
    train_dataset=CBCTdataset(input_dir=database, mode="train",transform=img_transform)
    validation_dataset=CBCTdataset(input_dir=database, mode="val",transform=img_transform)
    test_dataset=CBCTdataset(input_dir=database, mode="test",transform=img_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    ## Initialize Optimizer and Learning Rate Scheduler
    # optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # loss function 
    cost_bce = nn.BCEWithLogitsLoss()
    cost_dice = DICELoss()
    
    losses_train = []
    losses_val = []
    print("Start Training...")
    ## Start training 
    for epoch in tqdm(range(num_epochs)):
        model.train()
        loss_train = 0
        loss_val = 0
        for image, mask in train_dataloader:
            image = image.float()
            mask = mask.float()
            if display:
              display_data(image,mask)
              break
            optimizer.zero_grad()
            predictions= model(image.to(device))

            loss1 = cost_bce.forward(predictions[:,:,:,:,2],mask[:,:,:,:,2].to(device))
            # loss2 = cost_dice.forward(predictions,mask.to(device))
            # loss = loss1 + loss2
            # print(loss1,loss2)

            loss1.backward()
            optimizer.step()
            loss_train += loss1.item()
        # scheduler.step()
        # loss for each batch 
        losses_train.append(loss_train/ len(train_dataloader))
        if (epoch+1) % 10 == 0:
            print(losses_train[-1])

    # validation loss
        with torch.no_grad():
          model.eval()
          predictions = []
          for image, mask in validation_dataloader:
              image = image.float()
              mask = mask.float()
              predictions = model(image.to(device))

              loss1 = cost_bce.forward(predictions[:,:,:,:,2],mask[:,:,:,:,2].to(device))
              # loss2 = cost_dice.forward(predictions,mask.to(device))
              # loss = loss1 + loss2
              loss_val += loss1.item()

          losses_val.append(loss_val/ len(validation_dataloader))

        early_stopping(path,epoch,losses_train,losses_val, loss_val/ len(validation_dataloader), model,
                       )
        if early_stopping.early_stop:
            print('++++++++++Early Stopping...++++++++++')
            break
