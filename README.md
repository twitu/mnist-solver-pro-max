# MNIST Classification with PyTorch

![Tests](https://github.com/twitu/mnist-solver-pro-max/actions/workflows/model_tests.yml/badge.svg)

## Overview
This project implements a Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch. The model achieves >99.4% validation accuracy while maintaining a lightweight architecture (<20K parameters) through efficient use of modern deep learning techniques.

## Model Architecture
The network architecture incorporates several key features:
- Three sequential blocks with convolutional layers
- Batch Normalization for training stability
- Dropout (0.1) for regularization
- Global Average Pooling to reduce parameters
- 1x1 convolutions for channel dimension reduction

## Key Features
- Parameters: <20K
- Training Duration: 20 epochs
- Validation Accuracy: >99.4%
- Data Augmentation: Random rotation and affine transformations
- Learning Rate Scheduling: OneCycleLR policy

## Requirements
python
torch
torchvision
tqdm
matplotlib
torchsummary

## Usage
python
python script.py

## Model Performance
- The model achieves the target accuracy within the specified constraints
- Uses modern architectural choices to maintain efficiency
- Implements data augmentation for improved generalization

## Training Details
- Optimizer: Adam with weight decay
- Learning Rate: OneCycleLR scheduling
- Batch Size: 32
- Data Augmentation: 
  - Random rotation (±5 degrees)
  - Random affine transformation with shear
