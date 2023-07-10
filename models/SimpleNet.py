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
        self.features_output_dim = 32 * 1 * 1

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
