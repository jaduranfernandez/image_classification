import torch.nn as nn # basic building block for neural neteorks
import torch
import torch.nn.functional as F # import convolution functions like Relu
import os


class NeuralNetwork(nn.Module):
    """ 
    Models a simple Convolutional Neural Network with 2 Conv Layers & 2 FC
    """
    def __init__(self, filename = "model"):
        """ Initialize the neural network """
        self.model_name = filename
        super(NeuralNetwork, self).__init__()
        #   3 input image channel, 6 output channels,
        #   5x5 square convolution kernel (filter)
        self.conv1 = nn.Conv2d(3, 6, 5)
        #   Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) 
        #   Fully connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # 16 channels with 5x5 filters
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        ''' the forward propagation algorithm '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_model(self):        
        torch.save(self.model, self.model_name + ".pth")