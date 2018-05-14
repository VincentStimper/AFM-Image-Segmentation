# AFM Image Segmentation

Atomic Force Microscopy (AFM) images with DNA strands and nucleosome can be segmented by Fully Convolutional Neural Networks (FCN). This repository contains scripts to design, train and validate them.

## Network architecture

The network architecture is based on VGG16. The fully connected layers get transformed to convolutional layers. The features from these layers and the pooling layers get scored and upsampled. There are the architectures FCN-8, FCN-4, and FCN-2 where the number refers to the maximal upsampling factor.

## Requirments

The following packages are required:
 * tensorflow
 * numpy
 * scipy
 
 A pretrained version of VGG16 can be downloaded [here](ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy).