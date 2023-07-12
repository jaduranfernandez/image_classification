from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torch import randperm

class BaseDataLoader():
    """
    Base class for all data loaders
    """
    def __init__(self, root_folder, dataset, transform, batch_size, num_workers):

        self.trainset = dataset(root_folder, train = True, download = True, transform = transform)
        self.testset = dataset(root_folder, train = False, download = True, transform = transform)

        self.trainloader = DataLoader(self.trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
        self.testloader = DataLoader(self.testset, batch_size=batch_size,shuffle=False, num_workers=num_workers)
        self.classes = self.trainset.classes
        self.nClasses = len(self.classes)


class BaseSubsetLoader():
    """
    Base class for all data loaders
    """
    def __init__(self, root_folder, dataset, transform, batch_size, num_workers, subset_size):

        trainset = dataset(root_folder, train = True, download = True, transform = transform)
        self.testset = dataset(root_folder, train = False, download = True, transform = transform)

        subset_indices = randperm(len(trainset))[:subset_size]  # Índices aleatorios del subconjunto

        # Crear el subconjunto utilizando los índices aleatorios
        self.trainset = Subset(trainset, subset_indices)

        self.trainloader = DataLoader(self.trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
        self.testloader = DataLoader(self.testset, batch_size=batch_size,shuffle=False, num_workers=num_workers)
        self.classes = trainset.classes
        self.nClasses = len(self.classes)


