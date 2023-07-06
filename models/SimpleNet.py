import torch.nn as nn # basic building block for neural neteorks
import torch
import torch.nn.functional as F # import convolution functions like Relu
import os


class SimpleNet(nn.Module):
    """ 
    This model consists of 5 layers
    - 13 convolution layers with kernel 3x3
    - 3 fully connected layers
    - padding
    - 2x2 max pool
    - ReLu activation function
    """
    def __init__(self, filename = "SimpleNet", n_classes = 10):
        """ Initialize the neural network """
        self.model_name = filename
        super(SimpleNet, self).__init__()


        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(6, 16, kernel_size = 5, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),                
            nn.Conv2d(16, 32, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),                    
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 1 * 1, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(84, n_classes)
        )

    def forward(self, x):
        """ the forward propagation algorithm """        
        x = self.features(x)
        x = x.view(-1, 32 * 1 * 1)
        x = self.classifier(x)
        return x

    def save_model(self):        
        torch.save(self, self.model_name + ".pth")