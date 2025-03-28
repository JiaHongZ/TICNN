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
class DatasetCaltech101(data.Dataset):
    """
    # -----------------------------------------
    # model  train test val
    """
    def __init__(self, root=r'/zjh/data/caltech101_scale', scales=[1], model='train', transform=None):
        super(DatasetCaltech101, self).__init__()
        self.paths = []
        for scale in scales:
            path = read_image_data_from_csv(r'/zjh/NNA/data/caltech101/image_data_'+model+'_scale_'+str(scale)+'.csv')
            self.paths.extend(path)
        self.transform = transform
        # self.transform_lab = transforms.ToTensor()

    def __getitem__(self, index):
        # ------------------------------------
        # get H image
        # ------------------------------------
        image_path = self.paths[index][0]
        scale = float(self.paths[index][1])
        label = int(self.paths[index][2])

        image = util.imread_uint(image_path, 3)
        image = Image.fromarray(image, mode='RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image, label, scale

    def __len__(self):
        return len(self.paths)

def getcaltech101(data_dir, batch_size, usenormal=False):
    print('usenormal',usenormal)
    if usenormal:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((230, 230)),
                transforms.RandomRotation(15, ),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((230, 230)),
                transforms.RandomRotation(15, ),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
        }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    print(dataset_sizes)
    return  dataloaders, class_names

def getcaltech101_scale(data_dir, batch_size, usenormal=True):
    print('usenormal',usenormal)
    if usenormal:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomRotation(15, ),
                # transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomRotation(15, ),
                # transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
        }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    print(dataset_sizes)
    return  dataloaders, class_names

if __name__ == '__main__':
    path = r'/zjh/data/caltech101_scale'
    batch_size = 64
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]),
    }
    testset = DatasetCaltech101(path, [0.5,1.0], 'test', data_transforms['test'])
    train_loader = DataLoader(testset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=8,
                              drop_last=True,
                              pin_memory=True)

    from torchvision.transforms import ToPILImage
    import torchvision.models as models
    show = ToPILImage()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    for i, data in enumerate(train_loader):
        img, lab, scale = data
        print(img.shape, lab, scale)
        img_recon = (show(img[0].cpu()))
        plt.imshow(img_recon)
        plt.show()
        break
