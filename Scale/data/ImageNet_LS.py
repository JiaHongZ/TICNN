'''
We write this code with the help of PyTorch demo:
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

Dataset is Downloaded from https://www.kaggle.com/huangruichu/caltech101/version/2

Effects:
        transforms.Resize((230,230)),
        transforms.RandomRotation(15,),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),

wide_resnet101_2 SpinalNet_VGG gives 96.87% test accuracy

wide_resnet101_2 SpinalNet_ResNet gives 96.40% test accuracy
'''

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
import data.utils_image as util
import os
import copy
import pickle
from PIL import Image
from torch.utils.data import DataLoader
import csv

plt.ion()  # interactive mode
def read_image_data_from_csv(csv_filename):
    image_data = []
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 跳过第一行，即表头
        for row in reader:
            image_data.append(row)
    return image_data

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class DatasetImageNet1:
    """
    # -----------------------------------------
    # model  train test val
    """
    def __init__(self, root=r'/zjh/data/imagenet_scale', scales=[1], model='train', transform=None):
        if model == 'train':
            self.imagenet_val_dataset = datasets.ImageFolder(
                '/zjh/data/imagenet_scale/scale_'+str(scales[0])+'/train',  
                transform=transform
            )
        else:
            self.imagenet_val_dataset = datasets.ImageFolder(
                '/zjh/data/imagenet_scale/scale_'+str(scales[0])+'/val',  
                transform=transform
            )
            