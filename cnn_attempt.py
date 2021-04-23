import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from datetime import datetime

# contains all the classes in used in CIFAR10
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # inherits from the super
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5))
        # defining our first Convolutional layer
        # in_channels = how many layers does starting image have. We're RGB so its 3
        # out_channels = how many output dimensions do we want
        # kernel_size = the dimensions of the kernel we want to use
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # pooling layer, kernel size is 2x2 and the stride is 2 so the kernel won't look at the same pixels twice
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))
        # defining our second Convolutional layer with the input channels the same as the first layers output so they
        # can feed from one into the other.
        self.Relu = nn.ReLU()
        # carries out Relu
        self.normal_layer1 = nn.BatchNorm2d(32)
        # carries out Zero-centre and normalizes the output of the Convolutional layer before activation
        self.normal_layer2 = nn.BatchNorm2d(64)
        # carries out Zero-centre and normalizes the output of the Convolutional layer before activation
        self.linear_layer = nn.Linear(1600, len(classes))
        # output of linear layer is each class

    def forward(self, output):
        output = self.conv_layer1(output)
        output = self.normal_layer1(output)
        output = self.Relu(output)
        output = self.pooling(output)
        # repeat again but using second Convolutional layer now
        output = self.conv_layer2(output)
        output = self.normal_layer2(output)
        output = self.Relu(output)
        output = self.pooling(output)
        # second Convolutional layer complete
        output = output.reshape(output.size(0), -1)
        # we must reshape the output so it can be used by our linear layer
        output = self.linear_layer(output)
        # linear layer complete
        return output


if __name__ == '__main__':
    device = torch.device('cuda')
    # sets the default device as cuda so it will train on the GPU
    batch_size = 4
    # the amount of images to use per batch

    training_set = CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    # train_set contains the CIFAR10 training images, 50000 images, these images have been transformed into tensors
    # it will download the dataset if needed
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)
    # train loader contains groups of training images randomly shuffled in groups of 4 (the batch size)

    testing_set = CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
    # train_set contains the CIFAR10 testing images, 10000 images, these images have been transformed into tensors
    testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False, num_workers=2)
    # test loader contains groups of testing images randomly shuffled in groups of 4 (the batch size)

    classes = training_set.classes
    # print(len(training_set))

    loss_function = nn.CrossEntropyLoss()
    learning_rate = 0.001
    ConvolutionalNeuralNetwork = ConvolutionalNeuralNetwork()
    ConvolutionalNeuralNetwork.cuda()
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print(torch.cuda.memory_allocated())
    # initializing the Convolutional Neural Network
    optimizer = torch.optim.SGD(ConvolutionalNeuralNetwork.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer using stochastic gradient decent


    #train the dataset

    for loops in range(20):  # how many loops over the data you want to do
        print(datetime.now().strftime("%H:%M:%S"))
        for training_data in training_loader:
            inputs, labels = training_data
            inputs, labels = inputs.cuda(), labels.cuda()
            # gets input values and labels

            optimizer.zero_grad()
            # zeros the gradients

            outputs = ConvolutionalNeuralNetwork(inputs)

            # generates outputs

            loss = loss_function(outputs, labels)
            # applies loss function to outputs
            loss.backward()
            # generates the gradients (back propagation)
            optimizer.step()
            # applies stochastic gradient decent, so only a small sample is used to save time
    print(torch.cuda.memory_allocated())
    save_location = './cifar_net.pth'
    torch.save(ConvolutionalNeuralNetwork.state_dict(), save_location)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testing_loader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = ConvolutionalNeuralNetwork(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))