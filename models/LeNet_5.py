import torch.nn as nn # basic building block for neural neteorks
import torch
import torch.nn.functional as F # import convolution functions like Relu
import os


class LeNet_5(nn.Module):
    """ 
    This model consists of 5 layers
    - 2 convolution layers with kernel 5x5
    - 3 fully connected layers
    - padding
    - 2x2 average pool
    - tanh activation layer except the last(softmax)
    """
    def __init__(self, filename = "model", n_inputs = 1, n_outputs = 6, n_classes = 10):
        """ Initialize the neural network """
        self.model_name = filename
        super(LeNet_5, self).__init__()

        #   Convolution layers

        self.features = nn.Sequential(
            nn.Conv2d(n_inputs, n_outputs, 5, stride=1, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(n_outputs, 16, kernel_size=5, stride=1, padding = 1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 8, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Softmax(),
            nn.Linear(84, n_classes)
        )

    def forward(self, x):
        """ the forward propagation algorithm """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def save_model(self):        
        torch.save(self.model, self.model_name + ".pth")