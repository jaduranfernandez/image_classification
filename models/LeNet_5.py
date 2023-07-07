import torch.nn as nn # basic building block for neural neteorks
from base import BaseModel

class LeNet_5(BaseModel):
    """ 
    This model consists of 5 layers
    - 2 convolution layers with kernel 5x5
    - 3 fully connected layers
    - padding
    - 2x2 average pool
    - tanh activation layer except the last(softmax)
    """
    def __init__(self, filename = "LeNet_5", n_inputs = 3, n_outputs = 6, n_classes = 10):
        """ Initialize the neural network """
        super(LeNet_5, self).__init__()

        self.model_name = filename
        self.features_output_dim = 16 * 6 * 6

        self.features = nn.Sequential(
            nn.Conv2d(n_inputs, n_outputs, 5, stride=1, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(n_outputs, 16, kernel_size=5, stride=1, padding = 1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.features_output_dim, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Softmax(),
            nn.Linear(84, n_classes)
        )
