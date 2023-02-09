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








def normalize(img):
    return (img-img.min())/(img.max()-img.min())

def dice_score_image(prediction, target, n_classes):

    smooth = 1e-7
    dice_classes = np.zeros(n_classes)

    prediction = prediction.flatten(start_dim=2, end_dim=3)
    target = target.flatten(start_dim=2, end_dim=3)
    for cl in range(n_classes):

        TP = (prediction[:,cl,:]*target[:,cl,:]).sum(dim=1)
        DEN = (prediction[:,cl,:] + target[:,cl,:]).sum(dim = 1)

        dice_classes[cl] = ((2 * TP + smooth)/ (DEN+smooth)).mean()
    return dice_classes.mean()

def dice_score_dataset(model, dataloader, use_gpu=False):
    ## Number of Batches and Cache over Dataset
    n_batches = len(dataloader)
    scores = np.zeros(n_batches)
    ## Evaluate
    model.eval()
    idx = 0
    for data in dataloader:
        ## Format Data
        img, target = data
        img = img.float()
        target = target.float()
        if use_gpu:
            img = img.cuda()
            target = target.cuda()
        ## Make Predictions
        out = model(img)
        n_classes = out.shape[1]
        # prediction = torch.argmax(out, dim=1)
        # out = nn.Sigmoid()(out)
        out = normalize(out)

        prediction = out>0.5

        scores[idx] = dice_score_image(prediction, target, n_classes)
        idx += 1
    ## Average Dice Score Over Images

    m_dice = scores.mean()
    return m_dice



def evaluate(model, path, dataset, img_size = (96,96)):
    # without random crop to remove randomness
    img_transform = transforms.Compose([transforms.Resize(img_size),
        transforms.RandomVerticalFlip(0),transforms.RandomHorizontalFlip(0)
    ])

    device = torch.device('cuda')
    train_batch_size = 1
    validation_batch_size = 1

    ## Initialize Dataloaders
    train_dataset=CBCTdataset(input_dir=database, mode="train",transform=img_transform)
    validation_dataset=CBCTdataset(input_dir=database, mode="val",transform=img_transform)
    test_dataset=CBCTdataset(input_dir=database, mode="test",transform=img_transform)
    # shuffle = False for comparison
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if dataset == 'train':
        dataloader = train_dataloader
    elif dataset == 'validation':
        dataloader = validation_dataloader
    else:
        dataloader = test_dataloader

    checkpoint = torch.load( drive / path)
    model_trained = model(nclass=3).to(device)
    model_trained.load_state_dict(checkpoint['model'])
    losses_train = checkpoint['losses_train']
    losses_val = checkpoint['losses_val']
    # print(model_pretrain.oput)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(losses_train, 'C0')
    ax.set_ylabel('Loss',fontsize = 18)
    # ax2 = ax.twinx()
    ax.plot(losses_val, 'C1')
    ax.legend(['Train loss','Validation loss'],fontsize = 16)
    # ax.set_ylabel('Validation Loss', c='C1', fontweight='bold')
    # ax2.spines['right'].set_position(('axes', 1 + 0.3))
    ax.set_xlabel('Epoch',fontsize = 18)
    ax.set_title('Losses w.r.t epoch',fontweight='bold', fontsize = 20)
    plt.show()

    nimages = 14
    fig, axs = plt.subplots(nrows=3, ncols=nimages, figsize=(nimages*3,3*3), subplot_kw={'xticks': [], 'yticks': []})
    axs[0,0].set_ylabel('image')
    axs[1,0].set_ylabel('mask')
    axs[2,0].set_ylabel('Prediction')

    prec_list_ = []
    acc_list_ = []
    recall_list_ = []
    auc_list_ = []
    i = 0
    with torch.no_grad():
        model_trained.eval()
        for images, masks in dataloader: 
            if i >= nimages:
                break
            images = images.float()
            masks = masks.float()
            predictions = model_trained(images.to(device))
            for image, mask, prediction in zip(images, masks, predictions):
                # if i >= nimages:
                #     break
                image = image.numpy().transpose(1,2,0)
                truth = mask.cpu().numpy().reshape([-1])
                truth_ = []
                truth_.append(mask[0,:,:].cpu().numpy().reshape([-1]))
                truth_.append(mask[1,:,:].cpu().numpy().reshape([-1]))
                truth_.append(mask[2,:,:].cpu().numpy().reshape([-1]))
                
                for idx,each in enumerate(truth_):
                    truth_[idx] = each.astype('uint8')

                truth = truth.astype('uint8')

                mask = mask.numpy().transpose(1,2,0)

                prediction = normalize(prediction)

                pred = prediction.cpu().numpy().reshape([-1])
                fpr, tpr, threshold = roc_curve(truth,pred,pos_label=1)
                auc_img = auc(fpr,tpr)

                rgb_prediction = np.zeros((prediction.shape[1],prediction.shape[2],3))
                for c in range(3):
                    rgb_prediction[:,:,c] = prediction.cpu().numpy()[c,:,:]>0.5

                ac = []
                pc = []
                rc = []
                for c in range(3):
                    pred_cut = rgb_prediction[:,:,c].reshape([-1])
                    ac.append(accuracy_score(truth_[c],pred_cut))
                    pc.append(precision_score(truth_[c],pred_cut))
                    rc.append(recall_score(truth_[c],pred_cut))
                # print(pc,rc)
                acc_img = np.mean(ac)
                precisions = np.mean(pc)
                recalls = np.mean(rc)


                prec_list_.append(precisions)
                recall_list_.append(recalls)
                acc_list_.append(acc_img)
                auc_list_.append(auc_img)

                axs[0,i].imshow(image)
                axs[1,i].imshow(mask)
                axs[2,i].imshow(rgb_prediction)
                i += 1

    #Dice Score
    mdice = dice_score_dataset(model_trained, train_dataloader, use_gpu=True)
    print('train_dice score',mdice)
    mdice = dice_score_dataset(model_trained, test_dataloader, use_gpu=True)
    print('test_dice score',mdice)

    print('precision of test set is ',np.mean(prec_list_))
    print('recall of test set is ',np.mean(recall_list_))
    print('accuracy of test set is ',np.mean(acc_list_))
    print('auc of test set is ',np.mean(auc_list_))




def normalize(img):
    return (img-img.min())/(img.max()-img.min())

def dice_score_image_att(prediction, target, n_classes):

    smooth = 1e-7
    dice_classes = np.zeros(n_classes)

    prediction = prediction.flatten(start_dim=2, end_dim=3)
    target = target.flatten(start_dim=2, end_dim=3)
    for cl in range(n_classes):

        TP = (prediction[:,cl,:]*target[:,cl,:]).sum(dim=1)
        DEN = (prediction[:,cl,:] + target[:,cl,:]).sum(dim = 1)

        dice_classes[cl] = ((2 * TP + smooth)/ (DEN+smooth)).mean()
    return dice_classes.mean()

def dice_score_dataset_att(model, dataloader, use_gpu=False):
    ## Number of Batches and Cache over Dataset
    n_batches = len(dataloader)
    scores = np.zeros(n_batches)
    ## Evaluate
    model.eval()
    idx = 0
    for data in dataloader:
        ## Format Data
        img, target = data
        img = img.float()
        target = target.float()
        if use_gpu:
            img = img.cuda()
            target = target.cuda()
        ## Make Predictions
        out,att_map = model(img)
        n_classes = out.shape[1]
        # prediction = torch.argmax(out, dim=1)
        # out = nn.Sigmoid()(out)
        out = normalize(out)

        prediction = out>0.5

        scores[idx] = dice_score_image(prediction, target, n_classes)
        idx += 1
    ## Average Dice Score Over Images

    m_dice = scores.mean()
    return m_dice



def evaluate_att(model, path, dataset, img_size = (96,96)):
    # without random crop to remove randomness
    img_transform = transforms.Compose([transforms.Resize(img_size),
        transforms.RandomVerticalFlip(0),transforms.RandomHorizontalFlip(0)
    ])

    device = torch.device('cuda')
    train_batch_size = 1
    validation_batch_size = 1

    ## Initialize Dataloaders
    train_dataset=CBCTdataset(input_dir=database, mode="train",transform=img_transform)
    validation_dataset=CBCTdataset(input_dir=database, mode="val",transform=img_transform)
    test_dataset=CBCTdataset(input_dir=database, mode="test",transform=img_transform)
    # shuffle = False for comparison
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if dataset == 'train':
        dataloader = train_dataloader
    elif dataset == 'validation':
        dataloader = validation_dataloader
    else:
        dataloader = test_dataloader

    checkpoint = torch.load( drive / path)
    model_trained = model(nclass=3).to(device)
    model_trained.load_state_dict(checkpoint['model'])
    losses_train = checkpoint['losses_train']
    losses_val = checkpoint['losses_val']
    # print(model_pretrain.oput)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(losses_train, 'C0')
    ax.set_ylabel('Loss',fontsize = 18)
    # ax2 = ax.twinx()
    ax.plot(losses_val, 'C1')
    ax.legend(['Train loss','Validation loss'],fontsize = 16)
    # ax.set_ylabel('Validation Loss', c='C1', fontweight='bold')
    # ax2.spines['right'].set_position(('axes', 1 + 0.3))
    ax.set_xlabel('Epoch',fontsize = 18)
    ax.set_title('Losses w.r.t epoch',fontweight='bold', fontsize = 20)
    plt.show()

    nimages = 14
    fig, axs = plt.subplots(nrows=4, ncols=nimages, figsize=(nimages*3,3*3), subplot_kw={'xticks': [], 'yticks': []})
    axs[0,0].set_ylabel('image')
    axs[1,0].set_ylabel('mask')
    axs[2,0].set_ylabel('Prediction')
    axs[3,0].set_ylabel('Attention_map')

    prec_list_ = []
    acc_list_ = []
    recall_list_ = []
    auc_list_ = []
    i = 0
    with torch.no_grad():
        model_trained.eval()
        for images, masks in dataloader: 
            if i >= nimages:
                break
            images = images.float()
            masks = masks.float()
            predictions,att_maps = model_trained(images.to(device))
            for image, mask, prediction,att_map in zip(images, masks, predictions,att_maps):
                # if i >= nimages:
                #     break
                image = image.numpy().transpose(1,2,0)
                truth = mask.cpu().numpy().reshape([-1])
                truth_ = []
                truth_.append(mask[0,:,:].cpu().numpy().reshape([-1]))
                truth_.append(mask[1,:,:].cpu().numpy().reshape([-1]))
                truth_.append(mask[2,:,:].cpu().numpy().reshape([-1]))
                
                for idx,each in enumerate(truth_):
                    truth_[idx] = each.astype('uint8')

                truth = truth.astype('uint8')
                mask = mask.numpy().transpose(1,2,0)
                att_map = att_map.squeeze().cpu().detach().numpy()
                # att_map = att_map / att_map.max()

                prediction = normalize(prediction)
                pred = prediction.cpu().numpy().reshape([-1])
                fpr, tpr, threshold = roc_curve(truth,pred,pos_label=1)
                auc_img = auc(fpr,tpr)
                rgb_prediction = np.zeros((prediction.shape[1],prediction.shape[2],3))
                rgb_att_map = np.zeros((96,96,3))

                for c in range(3):
                    rgb_prediction[:,:,c] = prediction.cpu().numpy()[c,:,:]>0.5

                for j in range(3):
                    rgb_att_map[:,:,j] = att_map

                att_map_img = rgb_att_map*image
                att_map_img = (att_map_img-att_map_img.min())\
                /(att_map_img.max()-att_map_img.min())

                ac = []
                pc = []
                rc = []
                for c in range(3):
                    pred_cut = rgb_prediction[:,:,c].reshape([-1])
                    ac.append(accuracy_score(truth_[c],pred_cut))
                    pc.append(precision_score(truth_[c],pred_cut))
                    rc.append(recall_score(truth_[c],pred_cut))
                # print(pc,rc)
                acc_img = np.mean(ac)
                precisions = np.mean(pc)
                recalls = np.mean(rc)


                prec_list_.append(precisions)
                recall_list_.append(recalls)
                acc_list_.append(acc_img)
                auc_list_.append(auc_img)

                axs[0,i].imshow(image)
                axs[1,i].imshow(mask)
                axs[2,i].imshow(rgb_prediction)
                axs[3,i].imshow(att_map, cmap = 'coolwarm')
                # axs[3,i].imshow(att_map_img, cmap = 'coolwarm')


                i += 1

    #Dice Score
    mdice = dice_score_dataset_att(model_trained, train_dataloader, use_gpu=True)
    print('train_dice score',mdice)
    mdice = dice_score_dataset_att(model_trained, test_dataloader, use_gpu=True)
    print('test_dice score',mdice)

    print('precision of test set is ',np.mean(prec_list_))
    print('recall of test set is ',np.mean(recall_list_))
    print('accuracy of test set is ',np.mean(acc_list_))
    print('auc of test set is ',np.mean(auc_list_))



def normalize(img):
    return (img-img.min())/(img.max()-img.min())

def dice_score_image(prediction, target, n_classes):

    smooth = 1e-7
    dice_classes = np.zeros(n_classes)
    
    prediction = prediction.flatten(start_dim=2, end_dim=3)
    target = target.flatten(start_dim=2, end_dim=3)
    for cl in range(n_classes):

        TP = (prediction[:,cl,:]*target[:,cl,:]).sum(dim=1)
        DEN = (prediction[:,cl,:] + target[:,cl,:]).sum(dim = 1)

        dice_classes[cl] = ((2 * TP + smooth)/ (DEN+smooth)).mean()
    return dice_classes.mean()

def dice_score_dataset(model, dataloader, use_gpu=False):
    ## Number of Batches and Cache over Dataset
    n_batches = len(dataloader)
    scores = np.zeros(n_batches)
    ## Evaluate
    model.eval()
    idx = 0
    for data in dataloader:
        ## Format Data
        img, target = data
        img = img.float()
        target = target.float()
        if use_gpu:
            img = img.cuda()
            target = target.cuda()
        ## Make Predictions
        out = model(img)
        n_classes = out.shape[1]
        # prediction = torch.argmax(out, dim=1)
        # out = nn.Sigmoid()(out)
        out = (out[:,:,:,:,2])

        prediction = out>0.5

        scores[idx] = dice_score_image(prediction, target[:,:,:,:,2], n_classes)
        idx += 1
    ## Average Dice Score Over Images

    m_dice = scores.mean()
    return m_dice



def evaluate_3d(model, path, dataset, img_size = (96,96)):
    # without random crop to remove randomness
    img_transform = transforms.Compose([transforms.Resize(img_size),
        transforms.RandomVerticalFlip(0),transforms.RandomHorizontalFlip(0)
    ])

    device = torch.device('cuda')
    train_batch_size = 1
    validation_batch_size = 1

    ## Initialize Dataloaders
    train_dataset=CBCTdataset(input_dir=database, mode="train",transform=img_transform)
    validation_dataset=CBCTdataset(input_dir=database, mode="val",transform=img_transform)
    test_dataset=CBCTdataset(input_dir=database, mode="test",transform=img_transform)
    # shuffle = False for comparison
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if dataset == 'train':
        dataloader = train_dataloader
    elif dataset == 'validation':
        dataloader = validation_dataloader
    else:
        dataloader = test_dataloader

    checkpoint = torch.load( drive / path)
    model_trained = model(nclass=3).to(device)
    model_trained.load_state_dict(checkpoint['model'])
    losses_train = checkpoint['losses_train']
    losses_val = checkpoint['losses_val']
    # print(model_pretrain.oput)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(losses_train, 'C0')
    ax.set_ylabel('Loss',fontsize = 18)
    # ax2 = ax.twinx()
    ax.plot(losses_val, 'C1')
    ax.legend(['Train loss','Validation loss'],fontsize = 16)
    # ax.set_ylabel('Validation Loss', c='C1', fontweight='bold')
    # ax2.spines['right'].set_position(('axes', 1 + 0.3))
    ax.set_xlabel('Epoch',fontsize = 18)
    ax.set_title('Losses w.r.t epoch',fontweight='bold', fontsize = 20)
    plt.show()

    nimages = 14
    fig, axs = plt.subplots(nrows=3, ncols=nimages, figsize=(nimages*3,3*3), subplot_kw={'xticks': [], 'yticks': []})
    axs[0,0].set_ylabel('image')
    axs[1,0].set_ylabel('mask')
    axs[2,0].set_ylabel('Prediction')

    prec_list_ = []
    acc_list_ = []
    recall_list_ = []
    auc_list_ = []
    i = 0
    with torch.no_grad():
        model_trained.eval()
        for images, masks in dataloader: 
            if i >= nimages:
                break
            images = images.float()
            masks = masks.float()
            predictions = model_trained(images.to(device))
            for image, mask, prediction in zip(images, masks, predictions):
                # if i >= nimages:
                #     break
                image = image[:,:,:,2].numpy().transpose(1,2,0)
                truth = mask[:,:,:,2].cpu().numpy().reshape([-1])
                truth_ = []
                truth_.append(mask[0,:,:,2].cpu().numpy().reshape([-1]))
                truth_.append(mask[1,:,:,2].cpu().numpy().reshape([-1]))
                truth_.append(mask[2,:,:,2].cpu().numpy().reshape([-1]))
                
                for idx,each in enumerate(truth_):
                    truth_[idx] = each.astype('uint8')

                truth = truth.astype('uint8')

                mask = mask[:,:,:,2].numpy().transpose(1,2,0)

                # prediction = normalize(prediction[:,:,:,2])
                prediction = (prediction[:,:,:,2])


                pred = prediction.cpu().numpy().reshape([-1])
                fpr, tpr, threshold = roc_curve(truth,pred,pos_label=1)
                auc_img = auc(fpr,tpr)

                rgb_prediction = np.zeros((prediction.shape[1],prediction.shape[2],3))
                for c in range(3):
                    rgb_prediction[:,:,c] = prediction.cpu().numpy()[c,:,:]>0.5

                ac = []
                pc = []
                rc = []
                for c in range(3):
                    pred_cut = rgb_prediction[:,:,c].reshape([-1])
                    ac.append(accuracy_score(truth_[c],pred_cut))
                    pc.append(precision_score(truth_[c],pred_cut))
                    rc.append(recall_score(truth_[c],pred_cut))
                # print(pc,rc)
                acc_img = np.mean(ac)
                precisions = np.mean(pc)
                recalls = np.mean(rc)


                prec_list_.append(precisions)
                recall_list_.append(recalls)
                acc_list_.append(acc_img)
                auc_list_.append(auc_img)

                axs[0,i].imshow(image)
                axs[1,i].imshow(mask)
                axs[2,i].imshow(rgb_prediction)
                i += 1

    #Dice Score
    mdice = dice_score_dataset(model_trained, train_dataloader, use_gpu=True)
    print('train_dice score',mdice)
    mdice = dice_score_dataset(model_trained, test_dataloader, use_gpu=True)
    print('test_dice score',mdice)

    print('precision of test set is ',np.mean(prec_list_))
    print('recall of test set is ',np.mean(recall_list_))
    print('accuracy of test set is ',np.mean(acc_list_))
    print('auc of test set is ',np.mean(auc_list_))



def normalize(img):
    return (img-img.min())/(img.max()-img.min())

def dice_score_image(prediction, target, n_classes):

    smooth = 1e-7
    dice_classes = np.zeros(n_classes)
    
    prediction = prediction.flatten(start_dim=2, end_dim=3)
    target = target.flatten(start_dim=2, end_dim=3)
    for cl in range(n_classes):

        TP = (prediction[:,cl,:]*target[:,cl,:]).sum(dim=1)
        DEN = (prediction[:,cl,:] + target[:,cl,:]).sum(dim = 1)

        dice_classes[cl] = ((2 * TP + smooth)/ (DEN+smooth)).mean()
    return dice_classes.mean()

def dice_score_dataset(model, dataloader, use_gpu=False):
    ## Number of Batches and Cache over Dataset
    n_batches = len(dataloader)
    scores = np.zeros(n_batches)
    ## Evaluate
    model.eval()
    idx = 0
    for data in dataloader:
        ## Format Data
        img, target = data
        img = img.float()
        target = target.float()
        if use_gpu:
            img = img.cuda()
            target = target.cuda()
        ## Make Predictions
        out = model(img)
        n_classes = out.shape[1]
        # prediction = torch.argmax(out, dim=1)
        # out = nn.Sigmoid()(out)
        out = (out[:,:,:,:,2])

        prediction = out>0.5

        scores[idx] = dice_score_image(prediction, target[:,:,:,:,2], n_classes)
        idx += 1
    ## Average Dice Score Over Images

    m_dice = scores.mean()
    return m_dice



def evaluate_3d(model, path, dataset, img_size = (96,96)):
    # without random crop to remove randomness
    img_transform = transforms.Compose([transforms.Resize(img_size),
        transforms.RandomVerticalFlip(0),transforms.RandomHorizontalFlip(0)
    ])

    device = torch.device('cuda')
    train_batch_size = 1
    validation_batch_size = 1

    ## Initialize Dataloaders
    train_dataset=CBCTdataset(input_dir=database, mode="train",transform=img_transform)
    validation_dataset=CBCTdataset(input_dir=database, mode="val",transform=img_transform)
    test_dataset=CBCTdataset(input_dir=database, mode="test",transform=img_transform)
    # shuffle = False for comparison
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if dataset == 'train':
        dataloader = train_dataloader
    elif dataset == 'validation':
        dataloader = validation_dataloader
    else:
        dataloader = test_dataloader

    checkpoint = torch.load( drive / path)
    model_trained = model(nclass=3).to(device)
    model_trained.load_state_dict(checkpoint['model'])
    losses_train = checkpoint['losses_train']
    losses_val = checkpoint['losses_val']
    # print(model_pretrain.oput)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(losses_train, 'C0')
    ax.set_ylabel('Loss',fontsize = 18)
    # ax2 = ax.twinx()
    ax.plot(losses_val, 'C1')
    ax.legend(['Train loss','Validation loss'],fontsize = 16)
    # ax.set_ylabel('Validation Loss', c='C1', fontweight='bold')
    # ax2.spines['right'].set_position(('axes', 1 + 0.3))
    ax.set_xlabel('Epoch',fontsize = 18)
    ax.set_title('Losses w.r.t epoch',fontweight='bold', fontsize = 20)
    plt.show()

    nimages = 14
    fig, axs = plt.subplots(nrows=3, ncols=nimages, figsize=(nimages*3,3*3), subplot_kw={'xticks': [], 'yticks': []})
    axs[0,0].set_ylabel('image')
    axs[1,0].set_ylabel('mask')
    axs[2,0].set_ylabel('Prediction')

    prec_list_ = []
    acc_list_ = []
    recall_list_ = []
    auc_list_ = []
    i = 0
    with torch.no_grad():
        model_trained.eval()
        for images, masks in dataloader: 
            if i >= nimages:
                break
            images = images.float()
            masks = masks.float()
            predictions = model_trained(images.to(device))
            for image, mask, prediction in zip(images, masks, predictions):
                # if i >= nimages:
                #     break
                image = image[:,:,:,4].numpy().transpose(1,2,0)
                truth = mask[:,:,:,2].cpu().numpy().reshape([-1])
                truth_ = []
                truth_.append(mask[0,:,:,2].cpu().numpy().reshape([-1]))
                truth_.append(mask[1,:,:,2].cpu().numpy().reshape([-1]))
                truth_.append(mask[2,:,:,2].cpu().numpy().reshape([-1]))
                
                for idx,each in enumerate(truth_):
                    truth_[idx] = each.astype('uint8')

                truth = truth.astype('uint8')

                mask = mask[:,:,:,2].numpy().transpose(1,2,0)

                prediction = (prediction[:,:,:,2])

                pred = prediction.cpu().numpy().reshape([-1])
                fpr, tpr, threshold = roc_curve(truth,pred,pos_label=1)
                auc_img = auc(fpr,tpr)

                rgb_prediction = np.zeros((prediction.shape[1],prediction.shape[2],3))
                for c in range(3):
                    rgb_prediction[:,:,c] = prediction.cpu().numpy()[c,:,:]>0.5

                ac = []
                pc = []
                rc = []
                for c in range(3):
                    pred_cut = rgb_prediction[:,:,c].reshape([-1])
                    ac.append(accuracy_score(truth_[c],pred_cut))
                    pc.append(precision_score(truth_[c],pred_cut))
                    rc.append(recall_score(truth_[c],pred_cut))
                # print(pc,rc)
                acc_img = np.mean(ac)
                precisions = np.mean(pc)
                recalls = np.mean(rc)


                prec_list_.append(precisions)
                recall_list_.append(recalls)
                acc_list_.append(acc_img)
                auc_list_.append(auc_img)

                axs[0,i].imshow(image)
                axs[1,i].imshow(mask)
                axs[2,i].imshow(rgb_prediction)
                i += 1

    #Dice Score
    mdice = dice_score_dataset(model_trained, train_dataloader, use_gpu=True)
    print('train_dice score',mdice)
    mdice = dice_score_dataset(model_trained, test_dataloader, use_gpu=True)
    print('test_dice score',mdice)

    print('precision of test set is ',np.mean(prec_list_))
    print('recall of test set is ',np.mean(recall_list_))
    print('accuracy of test set is ',np.mean(acc_list_))
    print('auc of test set is ',np.mean(auc_list_))
