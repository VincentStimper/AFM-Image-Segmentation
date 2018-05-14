# AFM Image Segmentation

Atomic Force Microscopy (AFM) images with DNA strands and nucleosome can be segmented by Fully Convolutional Neural Networks (FCN). This repository contains scripts to design, train and validate them.


## Network architecture

The network architecture is based on [[1]](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). Take VGG16 and transform the fully connected layers to convolutional layers. The features from these layers and the pooling layers get scored and upsampled. There are the architectures FCN-8, FCN-4, and FCN-2 where the number refers to the maximal upsampling factor.  
The code was inspired by [[2]](https://github.com/MarvinTeichmann/tensorflow-fcn).


## Requirments

The code was developed for and tested with Python 3.6.

The following packages are required:
 * tensorflow
 * numpy
 * scipy  
 
 To install them run
 ```
pip3 install --upgrade tensorflow-gpu
pip3 install numpy scipy
```
 
 A pretrained version of VGG16 can be downloaded [here](https://goo.gl/vfvQi2).
 
 
 ## Usage
 
 The files `fcn8_vgg.py`, `fcn4_vgg.py`, and `fcn2_vgg.py` contain the class to create and build the models. This is done like:
 ```
 vgg_fcn = fcn8_vgg.FCN8VGG()  
 vgg_fcn.build(images)
 ```
 The file `utils.py` contains a function to save the network predictions as an image. To train the network run
 ```
python3 train_fcn8_vgg.py
```
 
 
 ## References
 
 [1] Jonathan Long, Evan Shelhamer, Trevor Darrell: Fully Convolutional Networks for Semantic Segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 39, Issue 4, 2016.  
 [2] [https://github.com/MarvinTeichmann/tensorflow-fcn](
 https://github.com/MarvinTeichmann/tensorflow-fcn), retrieved May 14, 2018.