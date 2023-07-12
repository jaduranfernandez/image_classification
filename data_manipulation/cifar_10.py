from base import BaseDataLoader, BaseSubsetLoader
from torchvision import datasets, transforms


def cifar10_transform():
    transform = transforms.Compose( # composing several transforms together
        [transforms.Resize((32,32)),
        transforms.ToTensor(), # to tensor object
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5
    return transform

class CifarDataLoader(BaseDataLoader):
    """
    Used to download data (if neccesary), load it, and prepare it for trainning and testing the model
    """
    def __init__(self, root_folder = './data', batch_size = 4, num_workers = 1):
        transform = transforms.Compose( # composing several transforms together
            [transforms.ToTensor(), # to tensor object
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5
        
        super().__init__(root_folder, datasets.CIFAR10, transform, batch_size, num_workers)
        self.classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class CifarSubsetLoader(BaseSubsetLoader):
    """
    Used to download data (if neccesary), load it, and prepare it for trainning and testing the model
    """
    def __init__(self, root_folder = './data', batch_size = 4, num_workers = 1):
        transform = transforms.Compose( # composing several transforms together
            [transforms.ToTensor(), # to tensor object
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5
        
        super().__init__(root_folder, datasets.CIFAR10, transform, batch_size, num_workers, subset_size=10000)
        self.classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
