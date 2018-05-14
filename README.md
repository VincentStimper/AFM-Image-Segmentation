# AFM Image Segmentation

Atomic Force Microscopy (AFM) images with DNA strands and nucleosome can be segmented by Fully Convolutional Neural Networks (FCN). This repository contains scripts to design, train and validate them.


## Network architecture

The network architecture is based on [1]. Take VGG16 and transform the fully connected layers to convolutional layers. The features from these layers and the pooling layers get scored and upsampled. There are the architectures FCN-8, FCN-4, and FCN-2 where the number refers to the maximal upsampling factor.  
The code was inspired by [2].


## Requirments

The following packages are required:
 * tensorflow
 * numpy
 * scipy
 
 A pretrained version of VGG16 can be downloaded [here](https://goo.gl/vfvQi2).
 
 
 ## References
 
 [1] Jonathan Long, Evan Shelhamer, Trevor Darrell: Fully Convolutional Networks for Semantic Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 39, Issue 4, 2016.  
 [2] [https://github.com/MarvinTeichmann/tensorflow-fcn](
 https://github.com/MarvinTeichmann/tensorflow-fcn), retrieved May 14, 2018.