import torch.nn as nn # basic building block for neural neteorks
from base import BaseModel


class SimpleNet(BaseModel):
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
        super(SimpleNet, self).__init__()
        self.model_name = filename
        self.features_output_dim = 128 * 4 * 4

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size = 2, stride = 2),            
        )        

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.features_output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, n_classes)
        )
