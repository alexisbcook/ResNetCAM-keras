[//]: # (Image References)

[image1]: ./images/dog_localization.png "Dog Localization"

# ResNetCAM-keras
Keras implementation of a ResNet-CAM model

## Motivation

The original Matlab implementation and paper (for AlexNet, GoogLeNet, and VGG16) can be found [here](https://github.com/metalbubble/CAM).  A Keras implementation of VGG-CAM can be found [here](https://github.com/tdeboissiere/VGG16CAM-keras/blob/master/README.md).

This implementation is written in Keras and uses ResNet-50, which was __not__ explored in the original paper.  

## Requirements

- keras with tensorflow backend (keras version 2.0.0 or later)
- numpy
- ast
- scipy
- matplotlib
- opencv3

## Usage

``` 
git clone https://github.com/alexisbcook/ResNetCAM-keras.git
cd ResNetCAM-keras
python ResNet_CAM.py images/dog.png
```

Feel free to try out your own image by replacing `images/dog.png` with a file path to another image! :)

## Example plots

![Dog Localization][image1]

## Blog Post

This repository is discussed in the blog post [here](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/).
