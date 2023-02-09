#%%
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from functools import reduce
import shutil
import random

data = np.load('your AnkleDataset_DS4.npz file path', allow_pickle=1)


def tcmap(fg):
    """Custom colormap with transparent background."""
    from matplotlib import colors
    fg = colors.to_rgb(fg)
    bg = colors.to_rgb((0,0,0))
    cmap = colors.LinearSegmentedColormap.from_list('binary', (bg,fg), 256)
    cmap._init()
    cmap._lut[:,-1] = np.linspace(0, 1, cmap.N + 3)
    return cmap


def printDistinct(arr):
    '''
    There function can extract unique value in an array
    '''
    n = len(arr)
    arr.sort();
    label_list = []
 
    # Traverse the sorted array
    for i in range(n):
         
        # Move the index ahead while there are duplicates
        if(i < n-1 and arr[i] == arr[i+1]):
            while (i < n-1 and (arr[i] == arr[i+1])):
                i+=1;
             
 
        # print last occurrence of the current element
        else:
            label_list.append(arr[i])
    return label_list

special_images = np.linspace(8,27,num=(27-8 +1), dtype=int)

def common_slice3(plane,fibula,tibia,talus,image_number):
    if image_number in special_images:
        arr1 = printDistinct(np.where(fibula == 1)[plane])
        arr2 = printDistinct(np.where(tibia == 2)[plane])
        arr3 = printDistinct(np.where(talus == 4)[plane])
    else:
        arr1 = printDistinct(np.where(fibula == 1)[plane])
        arr2 = printDistinct(np.where(tibia == 1)[plane])
        arr3 = printDistinct(np.where(talus == 1)[plane])
    #common_array = np.intersect1d(arr1, arr2)
    #common = np.intersect1d(common_array,arr3)
    common = reduce(np.intersect1d, (arr1,arr2,arr3))

    return common

def common_slice2(plane,fibula,tibia,talus,image_number):
    if image_number in special_images:
        arr1 = printDistinct(np.where(fibula == 1)[plane])
        arr2 = printDistinct(np.where(tibia == 2)[plane])
        arr3 = printDistinct(np.where(talus == 4)[plane])
    else:
        arr1 = printDistinct(np.where(fibula == 1)[plane])
        arr2 = printDistinct(np.where(tibia == 1)[plane])
        arr3 = printDistinct(np.where(talus == 1)[plane])
    common1 = np.intersect1d(arr1, arr2)
    common2 = np.intersect1d(arr2,arr3)
    common3 = np.intersect1d(arr1,arr3)
    common = np.unique(np.concatenate((common1,common2,common3),0))

    return common

def common_slice1(plane,fibula,tibia,talus,image_number):
    if image_number in special_images:
        arr1 = printDistinct(np.where(fibula == 1)[plane])
        arr2 = printDistinct(np.where(tibia == 2)[plane])
        arr3 = printDistinct(np.where(talus == 4)[plane])
    else:
        arr1 = printDistinct(np.where(fibula == 1)[plane])
        arr2 = printDistinct(np.where(tibia == 1)[plane])
        arr3 = printDistinct(np.where(talus == 1)[plane])
    common = np.unique(np.concatenate((arr1,arr2,arr3),0))
    return common

def common_slice3d(plane,fibula,tibia,talus,image_number):
    if image_number in special_images:
        arr1 = printDistinct(np.where(fibula == 1)[plane])
        arr2 = printDistinct(np.where(tibia == 2)[plane])
        arr3 = printDistinct(np.where(talus == 4)[plane])
    else:
        arr1 = printDistinct(np.where(fibula == 1)[plane])
        arr2 = printDistinct(np.where(tibia == 1)[plane])
        arr3 = printDistinct(np.where(talus == 1)[plane])
    common = reduce(np.intersect1d, (arr1,arr2,arr3))

    #return common[0::2]
    return common


def program3():
    plane = 0
    saveimagepath = '/content/drive/MyDrive/DLMI_Final/Data/data_3/image/' # CHANGE PATH HERE
    savelabelpath = '/content/drive/MyDrive/DLMI_Final/Data/data_3/label/' # CHANGE PATH HERE
    for i in range(1,int(len(data.files)/4+1)):
        # assign file
        img = data['image_' + str(i)]
        fibula = data['fibula_' + str(i)]
        tibia = data['tibia_' + str(i)]
        talus = data['talus_' + str(i)]
        # defualt plane : sagittal
        slices = common_slice3(plane,fibula,tibia,talus,i)

        slices_idx = 1
        for slc in slices:
            if plane == 0:
                if i in special_images:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*75,talus[slc,:,:]*56.25))
                else:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*150,talus[slc,:,:]*225))
            else:
                if i in special_images:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*75,talus[slc,:,:]*56.25))
                else:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*150,talus[slc,:,:]*225))
            # img = np.expand_dims(img, axis=2)
            # print(img.shape)
            # print(label.shape)
            # set image and label name
            imgname = 'image'+str(i)+'_' + str(slices_idx)+'.npy'
            labelname = 'label'+str(i)+'_' + str(slices_idx)+'.npy'
            slices_idx += 1
            imgpath = saveimagepath + imgname
            labelpath = savelabelpath + labelname

            if image_my.shape[1] == 144:
                image_my = image_my[0:90, 40:140]
                label = label[0:90, 40:140,:]
            elif image_my.shape[1] == 120:
                image_my = image_my[10:100, 20:120]
                label = label[10:100, 20:120,:]
            elif image_my.shape[1] == 139:
                image_my = image_my[5:95, 20:120]
                label = label[5:95, 20:120,:]

            np.save(imgpath,image_my)
            np.save(labelpath,label)
    return




def program1():
    plane = 0
    saveimagepath = '/content/drive/MyDrive/DLMI_Final/Data/data_1/image/' # CHANGE PATH HERE
    savelabelpath = '/content/drive/MyDrive/DLMI_Final/Data/data_1/label/' # CHANGE PATH HERE
    for i in range(1,int(len(data.files)/4+1)):
        # assign file
        img = data['image_' + str(i)]
        fibula = data['fibula_' + str(i)]
        tibia = data['tibia_' + str(i)]
        talus = data['talus_' + str(i)]
        # defualt plane : sagittal
        slices = common_slice1(plane,fibula,tibia,talus,i)

        slices_idx = 1
        for slc in slices:
            slc = int(slc)
            if plane == 0:
                if i in special_images:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*75,talus[slc,:,:]*56.25))
                else:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*150,talus[slc,:,:]*225))
            else:
                if i in special_images:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*75,talus[slc,:,:]*56.25))
                else:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*150,talus[slc,:,:]*225))
            # img = np.expand_dims(img, axis=2)
            # print(img.shape)
            # print(label.shape)
            # set image and label name
            imgname = 'image'+str(i)+'_' + str(slices_idx)+'.npy'
            labelname = 'label'+str(i)+'_' + str(slices_idx)+'.npy'
            slices_idx += 1
            imgpath = saveimagepath + imgname
            labelpath = savelabelpath + labelname

            if image_my.shape[1] == 144:
                image_my = image_my[0:90, 40:140]
                label = label[0:90, 40:140,:]
            elif image_my.shape[1] == 120:
                image_my = image_my[10:100, 20:120]
                label = label[10:100, 20:120,:]
            elif image_my.shape[1] == 139:
                image_my = image_my[5:95, 20:120]
                label = label[5:95, 20:120,:]

            np.save(imgpath,image_my)
            np.save(labelpath,label)
    return
def program2():
    plane = 0
    saveimagepath = '/content/drive/MyDrive/DLMI_Final/Data/data_2/image/' # CHANGE PATH HERE
    savelabelpath = '/content/drive/MyDrive/DLMI_Final/Data/data_2/label/' # CHANGE PATH HERE
    for i in range(1,int(len(data.files)/4+1)):
        # assign file
        #print("Processing patient %d"%i)
        img = data['image_' + str(i)]
        fibula = data['fibula_' + str(i)]
        tibia = data['tibia_' + str(i)]
        talus = data['talus_' + str(i)]
        # defualt plane : sagittal
        slices = common_slice2(plane,fibula,tibia,talus,i)
        #print("Patient %d Common slices:\n")
        #print(slices)

        slices_idx = 1
        for slc in slices:
            if plane == 0:
                if i in special_images:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*75,talus[slc,:,:]*56.25))
                else:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*150,talus[slc,:,:]*225))
            else:
                if i in special_images:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*75,talus[slc,:,:]*56.25))
                else:
                    image_my = img[slc,:,:]
                    label = np.dstack((fibula[slc,:,:]*30,tibia[slc,:,:]*150,talus[slc,:,:]*225))
            # img = np.expand_dims(img, axis=2)
            # print(img.shape)
            # print(label.shape)
            # set image and label name
            imgname = 'image'+str(i)+'_' + str(slices_idx)+'.npy'
            labelname = 'label'+str(i)+'_' + str(slices_idx)+'.npy'
            slices_idx += 1
            imgpath = saveimagepath + imgname
            labelpath = savelabelpath + labelname

            if image_my.shape[1] == 144:
                image_my = image_my[0:90, 40:140]
                label = label[0:90, 40:140,:]
            elif image_my.shape[1] == 120:
                image_my = image_my[10:100, 20:120]
                label = label[10:100, 20:120,:]
            elif image_my.shape[1] == 139:
                image_my = image_my[5:95, 20:120]
                label = label[5:95, 20:120,:]

            np.save(imgpath,image_my)
            #print("%s saved"%imgpath)
            np.save(labelpath,label)
    return


def program_3d():
    plane = 0
    saveimagepath = '/content/drive/MyDrive/DLMI_Final/Data/thickened/image/' # CHANGE PATH HERE
    savelabelpath = '/content/drive/MyDrive/DLMI_Final/Data/thickened/label/' # CHANGE PATH HERE
    for i in range(1,int(len(data.files)/4+1)):
        # assign file
        img = data['image_' + str(i)]
        fibula = data['fibula_' + str(i)]
        tibia = data['tibia_' + str(i)]
        talus = data['talus_' + str(i)]
        # defualt plane : sagittal
        slices = common_slice3d(plane,fibula,tibia,talus,i)
        #print("Patient %d has targeted slices:"%i)
        #print(slices)
        slices_idx = 1
        for slc in slices:
            if plane == 0:
                if i in special_images:
                    image_my = img[slc-2:slc+3,:,:]
                    label = np.stack((fibula[slc-2:slc+3,:,:]*30,tibia[slc-2:slc+3,:,:]*75,talus[slc-2:slc+3,:,:]*56.25)).transpose([1,2,3,0])
                else:
                    image_my = img[slc-2:slc+3,:,:]
                    label = np.stack((fibula[slc-2:slc+3,:,:]*30,tibia[slc-2:slc+3,:,:]*150,talus[slc-2:slc+3,:,:]*225)).transpose([1,2,3,0])
            else:
                if i in special_images:
                    image_my = img[slc-2:slc+3-2:slc+2,:,:]
                    label = np.stack((fibula[slc-2:slc+3,:,:]*30,tibia[slc-2:slc+3,:,:]*75,talus[slc-2:slc+3,:,:]*56.25)).transpose([1,2,3,0])
                else:
                    image_my = img[slc-2:slc+3,:,:]
                    label = np.stack((fibula[slc-2:slc+3,:,:]*30,tibia[slc-2:slc+3,:,:]*150,talus[slc-2:slc+3,:,:]*225)).transpose([1,2,3,0])



            # img = np.expand_dims(img, axis=2)
            # print(img.shape)
            # print(label.shape)
            # set image and label name


            imgname = 'image'+str(i)+'_' + str(slices_idx)+'.npy'
            labelname = 'label'+str(i)+'_' + str(slices_idx)+'.npy'
            slices_idx += 1
            imgpath = saveimagepath + imgname
            labelpath = savelabelpath + labelname

            if image_my.shape[1] == 144:
                image_my = image_my[:,0:90, 40:140]
                label = label[:,0:90, 40:140,:]
            elif image_my.shape[1] == 120:
                image_my = image_my[:,10:100, 20:120]
                label = label[:,10:100, 20:120,:]
            elif image_my.shape[1] == 139:
                image_my = image_my[:,5:95, 20:120]
                label = label[:,5:95, 20:120,:]

            np.save(imgpath,image_my)
            np.save(labelpath,label)
    return

#%%
########### RUN THIS PART WILL GENERATE DATA
program3()
print("Data_3 generated")
program2()
print("Data_2 generated")
program_3d()
print("thickened data generated")














#%%
########### SPLIT DATA_3
database = "Change path here to data_3"
# data spliting 
filenames = os.listdir(database/'image')
filenames.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(18)
random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)
labelfiles = []
for filename in filenames:
    labelfiles.append("label"+filename[5:])

split1 = train = int(0.7 * len(filenames))
split2 = int(0.9 * len(filenames))

train_filenames = filenames[:split1]
dev_filenames = filenames[split1:split2]
test_filenames = filenames[split2:]

train_labelnames = labelfiles[:split1]
dev_labelnames = labelfiles[split1:split2]
test_labelnames = labelfiles[split2:]


for imagefile in train_filenames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_3/image/%s' % imagefile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_3/train/image/%s' % imagefile)  # CHANGE PATH HERE
for imagefile in dev_filenames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_3/image/%s' % imagefile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_3/valid/image/%s' % imagefile)  # CHANGE PATH HERE
for imagefile in test_filenames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_3/image/%s' % imagefile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_3/test/image/%s' % imagefile)  # CHANGE PATH HERE
for labelfile in  train_labelnames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_3/label/%s' % labelfile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_3/train/label/%s' % labelfile)  # CHANGE PATH HERE
for labelfile in  dev_labelnames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_3/label/%s' % labelfile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_3/valid/label/%s' % labelfile)  # CHANGE PATH HERE
for labelfile in  test_labelnames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_3/label/%s' % labelfile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_3/test/label/%s' % labelfile)  # CHANGE PATH HERE


#%%
###### SPLIT THICKENED DATA
database = "Change path here to thickened"
# data spliting 
filenames = os.listdir(database/'image')
filenames.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(18)
random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)
labelfiles = []
for filename in filenames:
    labelfiles.append("label"+filename[5:])

split1 = train = int(0.7 * len(filenames))
split2 = int(0.9 * len(filenames))

train_filenames = filenames[:split1]
dev_filenames = filenames[split1:split2]
test_filenames = filenames[split2:]

train_labelnames = labelfiles[:split1]
dev_labelnames = labelfiles[split1:split2]
test_labelnames = labelfiles[split2:]


# CHANGE PATH BELOW

for imagefile in train_filenames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/thickened/image/%s' % imagefile,
                './drive/MyDrive/DLMI_Final/Data/thickened/train/image/%s' % imagefile)
for imagefile in dev_filenames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/thickened/image/%s' % imagefile,
                './drive/MyDrive/DLMI_Final/Data/thickened/valid/image/%s' % imagefile)
for imagefile in test_filenames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/thickened/image/%s' % imagefile,
                './drive/MyDrive/DLMI_Final/Data/thickened/test/image/%s' % imagefile)
for labelfile in  train_labelnames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/thickened/label/%s' % labelfile,
                './drive/MyDrive/DLMI_Final/Data/thickened/train/label/%s' % labelfile)
for labelfile in  dev_labelnames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/thickened/label/%s' % labelfile,
                './drive/MyDrive/DLMI_Final/Data/thickened/valid/label/%s' % labelfile)
for labelfile in  test_labelnames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/thickened/label/%s' % labelfile,
                './drive/MyDrive/DLMI_Final/Data/thickened/test/label/%s' % labelfile)


#%%
####### SPLIT DATA_2

database = "Change path here to data_2"
# data spliting 
filenames = os.listdir(database/'image')
filenames.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(18)
random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)
labelfiles = []
for filename in filenames:
    labelfiles.append("label"+filename[5:])

split1 = train = int(0.7 * len(filenames))
split2 = int(0.9 * len(filenames))

train_filenames = filenames[:split1]
dev_filenames = filenames[split1:split2]
test_filenames = filenames[split2:]

train_labelnames = labelfiles[:split1]
dev_labelnames = labelfiles[split1:split2]
test_labelnames = labelfiles[split2:]


for imagefile in train_filenames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_2/image/%s' % imagefile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_2/train/image/%s' % imagefile)  # CHANGE PATH HERE
for imagefile in dev_filenames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_2/image/%s' % imagefile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_2/valid/image/%s' % imagefile)  # CHANGE PATH HERE
for imagefile in test_filenames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_2/image/%s' % imagefile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_2/test/image/%s' % imagefile)  # CHANGE PATH HERE
for labelfile in  train_labelnames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_2/label/%s' % labelfile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_2/train/label/%s' % labelfile)  # CHANGE PATH HERE
for labelfile in  dev_labelnames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_2/label/%s' % labelfile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_2/valid/label/%s' % labelfile)  # CHANGE PATH HERE
for labelfile in  test_labelnames:
    shutil.copy('./drive/MyDrive/DLMI_Final/Data/data_2/label/%s' % labelfile,  # CHANGE PATH HERE
                './drive/MyDrive/DLMI_Final/Data/data_2/test/label/%s' % labelfile)  # CHANGE PATH HERE