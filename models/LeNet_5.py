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
    def __init__(self, filename = "model", n_inputs = 3, n_outputs = 6, n_classes = 10):
        """ Initialize the neural network """
        self.model_name = filename
        super(LeNet_5, self).__init__()

        #   Convolution layers

        """ self.conv1 = nn.Conv2d(3, 6, 5, stride = 1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride = 1)
        self.pool = nn.AvgPool2d(2, 2)

        #   Fully connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # 16 channels with 5x5 filters
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
         """

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
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 1)
        x = self.fc3(x)

        return x

    def save_model(self):        
        torch.save(self.model, self.model_name + ".pth")