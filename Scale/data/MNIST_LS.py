import numpy as np
import h5py
from matplotlib import pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
from einops import rearrange, repeat

# MNIST Large Scale

def load_mnist_h5_tr_te_val(n_train, n_test, n_val, filename,
                            path):
    """Data will be in format [n_samples, width, height, n_channels] """

    with h5py.File(path + filename, 'r') as f:
        # Data should be floating point
        x_train = np.array(f["/x_train"], dtype=np.float32)
        x_test = np.array(f["/x_test"], dtype=np.float32)
        x_val = np.array(f["/x_val"], dtype=np.float32)

        # Labels should normally be integers
        y_train = np.array(f["/y_train"], dtype=np.int32)
        y_test = np.array(f["/y_test"], dtype=np.int32)
        y_val = np.array(f["/y_val"], dtype=np.int32)

        # Labels should normally be 1D vectors, shape (n_labels,)
        y_train = np.reshape(y_train, (np.size(y_train),))
        y_test = np.reshape(y_test, (np.size(y_test),))
        y_val = np.reshape(y_val, (np.size(y_val),))

    # Handle case of data containing only a single sample
    # (which is the case for the train and validation partitions in the "testdata only" datasets)
    if len(np.shape(x_train)) == 3:
        x_train = np.expand_dims(x_train, 0)
    if len(np.shape(x_test)) == 3:
        x_test = np.expand_dims(x_test, 0)
    if len(np.shape(x_val)) == 3:
        x_val = np.expand_dims(x_val, 0)

        # Possibly use a different number of samples than in the datasetfile
    x_train = x_train[0:n_train]
    x_test = x_test[0:n_test]
    x_val = x_val[0:n_val]

    y_train = y_train[0:n_train]
    y_test = y_test[0:n_test]
    y_val = y_val[0:n_val]

    assert np.shape(x_train)[0] == n_train
    assert np.shape(x_test)[0] == n_test
    assert np.shape(x_val)[0] == n_val

    assert np.shape(y_train)[0] == n_train
    assert np.shape(y_test)[0] == n_test
    assert np.shape(y_val)[0] == n_val

    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


def load_mnist_h5_te(n_test, filename, path):
    """Data will be in format [n_samples, width, height, n_channels] """

    with h5py.File(path + filename, 'r') as f:
        # Data should be floating point
        x_test = np.array(f["/x_test"], dtype=np.float32)
        # Labels should normally be integers
        y_test = np.array(f["/y_test"], dtype=np.int32)

        # Labels should normally be 1D vectors, shape (n_labels,)
        y_test = np.reshape(y_test, (np.size(y_test),))

    # Handle case of data containing only a single sample
    # (which is the case for the train and validation partitions in the "testdata only" datasets)
    if len(np.shape(x_test)) == 3:
        x_test = np.expand_dims(x_test, 0)

        # Possibly use a different number of samples than in the datasetfile
    x_test = x_test[0:n_test]
    y_test = y_test[0:n_test]
    assert np.shape(x_test)[0] == n_test
    assert np.shape(y_test)[0] == n_test

    return (x_test, y_test)
# path = ''

'''
mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr1p000_scte1p000.h5
mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr2p000_scte2p000.h5
mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr4p000_scte4p000.h5
mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr1p000-4p000_scte1p000-4p000.h5
'''

class MNIST_LS(Dataset):
    def __init__(self, path="E:\dataset\MNIST_Large_Scale\\", filename="mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr1p000_scte1p000.h5",
                 data_model='train', im_size=224, transform = None, n_train = 50000, n_val = 10000, n_test = 10000):
        super(Dataset, self).__init__()

        self.data_model = data_model
        if data_model == 'train':
            data = load_mnist_h5_tr_te_val(n_train, n_test, n_val, filename, path) # (x_train, y_train), (x_test, y_test), (x_val, y_val)
            (self.x_train, self.y_train), (x_test, y_test), (x_val, y_val) = data

        elif data_model == 'test':
            data = load_mnist_h5_te(n_test, filename, path) # (x_test, y_test)
            (self.x_test, self.y_test) = data
        else:
            raise RuntimeError("No data model:", data_model)
        self.transform = transform
        self.im_size = im_size
    def __getitem__(self, index):
        if self.data_model == 'train':
            if self.transform != None:
                x_train = self.transform(self.x_train[index])
            else:
                x_train = self.x_train[index]
            y_train = self.y_train[index]
                # x_val = self.transform(x_val)
                # x_test = self.transform(x_test)
            return (x_train, y_train)

        elif self.data_model == 'test':
            if self.transform != None:
                x_test = self.transform(self.x_test[index])
            else:
                x_test = self.x_test[index]
            y_test = self.y_test[index]
                # x_val = self.transform(x_val)
                # x_test = self.transform(x_test)
            return (x_test, y_test)

        else:
            raise RuntimeError("No data model:", self.data_model)

    def __len__(self):
        if self.data_model == 'train':
            return len(self.x_train)
        elif self.data_model == 'test':
            return len(self.x_test)