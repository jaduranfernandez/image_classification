from base.base_data_loader import BaseDataLoader
from torchvision import datasets, transforms



class CifarDataLoader(BaseDataLoader):
    """
    Used to download data (if neccesary), load it, and prepare it for trainning and testing the model
    """
    def __init__(self, root_folder = './data', batch_size = 4, shuffle = True, validation_split = 0.0, num_workers = 1, training = True):
        transform = transforms.Compose( # composing several transforms together
            [transforms.ToTensor(), # to tensor object
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5
        
        self.data_dir = root_folder
        self.dataset = datasets.CIFAR10(self.data_dir, train = training, download = True, transform = transform)
        self.classes = self.dataset.classes
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

