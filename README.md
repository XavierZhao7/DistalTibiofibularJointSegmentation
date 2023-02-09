# Distal tibiofibular joint segmentation in CT reconstructions

## Introduction

CT-reconstruction image segmentation is essential in the quantitative bone and joint surgery assessment of the distal tibiofibular joints. Multiple studies have developed methods for coping with the segmentation task. This study aimed to deliver an automatic segmentation method of cone-beam CT (CBCT) image in the analysis of Distal tibiofibular joints injury.
The U-Net[1] is a convolutional network architecture for fast and precise segmentation of images. Thus, U-Net-based architectures are commonly used in computer vision segmentation tasks. However, due to the downsampling operation, it lacks the ability to extract the delicate feature. The dice loss is designed to solve the problem of imbalanced data. The attention mechanism [2] is a technique that mimics cognitive attention, which enhances some parts of the input data while diminishing other parts of the image. 3DUNet [3] is proposed to leverage the information of neighboring slices in 3d volume.
In our work, the square version dice loss is used to replace the original binary cross entropy loss. We also develop an attention block that combines spatial and channel-wise relationships of the feature maps in different levels. In our case, both the 3d and 2d solution for 3DUNet are developed on our thickened CBCT dataset.
The dataset contains 40 3D CT-reconstruction images and ground truth segmentations of the fibula, tibia, and talus bones. After 2D / 3D image extraction and preprocessing, this project aims to improve the ability of UNet to detect distal tibiofibular joints by implementing attention mechanisms. Dice loss was introduced in the training process to prevent data imbalance. Dice score, AUC of Receiver operating characteristic, accuracy(ACC), precision(PRC), and Recall were included in the evaluation process to compare performances of the UNet, Dice-UNet, SCAUNet, 3D UNet and 3D UNet 2D on this dataset.

###...
Complete report is upon request

## Methodologies


Using U-Net[1] as our baseline, we first investigated how dataset design influences the model ability. The random vertical flip with probability equals to 0.5 is used as our basic data augmentation method. The horizontal flip does not make sense morphologically in this case. We then explore the angle selection for random rotation and the optimal 90 degrees are set experimentally. The elastic deformation in the original UNet paper is omitted because we have a fair amount of annotated data.
Since the labels for each 3 class are one-hot encoded and saved in 3 separate channels, the binary cross entropy (bce) is used as our basic loss function. The dice loss is then introduced to replace the bce loss to address the problem of data imbalance, i.e., the imbalance between foreground and background due to a). the different positions of the tibiofibular joint appearing in different slices and b) the variance of different patients.
The attention mechanism that includes an additional attention block into the vanilla UNet (see figure 3) is introduced to adjust the weight of the feature maps extracted by the U-Net encoder to focus on certain features while suppressing others. The spatial-channel attention embedded UNet(SCAUNet) includes 2 types of attention mechanisms: 1) The spatial attention utilizes the inter-spatial relationship of features and generates a spatial attention map by extracting a feature descriptor from each input channel and then applying a convolution layer on the descriptor. 2) Channel attention utilizes the inter-channel relationship of features and generates a channel attention map by extracting a feature descriptor across all channels and then applying a convolution layer on it. The generated spatial and channel-wise attention map are then applied ,by element-wise multiplication, to reweigh the feature maps out of the encoders in different levels.
Using the uncropped version of thickened data_3, the 3D UNet is designed to leverage the information of neighboring slices of data and reduce noise. The maxpooling3d and Convtranspose3d are used in encoder and decoder to downsample and upsample the x-axis and y-axis, while keeping the z-axis(slices) unchanged. The data augmentation method, which is the same as UNet, is applied slice by slice during training. The 3DUNet is a 3d solution that uses the 3d thickened label to calculate the 3d bce loss. While 3DUNet2d is a 2d solution that selects the center slice of the thickened image and matrix and calculates the 2d bce loss. The number of feature channels and down/up sampling layer remain the same as vanilla UNet and we did not encounter memory problems.

### Training methods

Both the cropped image and uncropped image are resized to 96×96 to fit the model input layer. For cross entropy loss, we first tune our hyperparameters to be optimal on the validation set of data_3. The same set-up is extended to data_2 and attention-embedded models. The UNet used dice loss(Dice-UNet) is trained with a smaller learning rate due to the unstable training process of them. Using SGD with momentum = 0.9 as optimizer, the learning rates for cross entropy loss and dice loss are 0.01 and 0.005, respectively. The batch_size is set to 5 for all models. All models are trained for 500 epochs with early-stopping implemented. The vanilla UNet, dice-UNet and SCAUNet(bce) using data_3 are trained and evaluated in 5 trials. Due to the computational resource and time limitation, data_2 is only trained once for UNet and SCAUNet. Also, the SCAUNet(dice) is trained only once. The 3DUNet and 3DUNet2d are trained once using the same hyperparameters as vanilla UNet.

### Evaluation

5 metrics: Dice score, Accuracy, AUC(ROC), Precision and Recall are used to evaluate the model performance on the test set. The mean, standard deviation and 95% confidence intervals (CI) of the trials are calculated.


### ...

Complete report is upon request
