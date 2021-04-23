import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')
# sets the default device as cuda so it will train on the GPU
batch_size = 4
# the amount of images to use per batch

train_set = CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor)
# train_set contains the CIFAR10 training images, 50000 images, these images have been transformed into tensors
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
# train loader contains groups of training images randomly shuffled in groups of 4 (the batch size)

test_set = CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor)
# train_set contains the CIFAR10 testing images, 10000 images, these images have been transformed into tensors
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
# test loader contains groups of testing images randomly shuffled in groups of 4 (the batch size)


classes = train_set.classes
print(len(train_set))


# contains all the classes in used in CIFAR10
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # inherits from the super
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3))
        # defining our first Convolutional layer
        # in_channels = how many layers does starting image have. We're RGB so its 3
        # out_channels = how many output dimensions do we want
        # kernel_size = the dimensions of the kernal we want to use
        self.conv_layer2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3))
        # defining our second Convolutional layer with the same channels as the first one