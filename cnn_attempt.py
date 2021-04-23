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

training_set = CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor)
# train_set contains the CIFAR10 training images, 50000 images, these images have been transformed into tensors
# it will download the dataset if needed
training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)
# train loader contains groups of training images randomly shuffled in groups of 4 (the batch size)

testing_set = CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor)
# train_set contains the CIFAR10 testing images, 10000 images, these images have been transformed into tensors
testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False, num_workers=2)
# test loader contains groups of testing images randomly shuffled in groups of 4 (the batch size)


classes = training_set.classes
print(len(training_set))


# contains all the classes in used in CIFAR10
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # inherits from the super
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3))
        # defining our first Convolutional layer
        # in_channels = how many layers does starting image have. We're RGB so its 3
        # out_channels = how many output dimensions do we want
        # kernel_size = the dimensions of the kernel we want to use

        self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # pooling layer, kernel size is 2x2 and the stride is 2 so the kernel won't look at the same pixels twice

        self.conv_layer2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3))
        # defining our second Convolutional layer with the input channels the same as the first layers output so they
        # can feed from one into the other.
