import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    ''' function to show image '''
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy() # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    return plt


class DataLoader():
    """
    Used to download data (if neccesary), load it, and prepare it for trainning and testing the model
    """
    def __init__(self, root_folder = './data', batch_size = 4, num_workers = 2):
        transform = transforms.Compose( # composing several transforms together
            [transforms.ToTensor(), # to tensor object
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5

        # load train data
        self.trainset = torchvision.datasets.CIFAR10(root = root_folder, train = True,
                                                download=True, transform=transform)
        
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size = batch_size,
                                                shuffle = True, num_workers = num_workers)

        # load test data
        self.testset = torchvision.datasets.CIFAR10(root = root_folder, train=False,
                                            download = True, transform = transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size = batch_size,
                                                shuffle = False, num_workers = num_workers)

        # put 10 classes into a set
        self.classes = self.trainset.classes

    def show_subplot(self, rows = 4, cols = 4):
        """
        Allows the user to see on a dimension specified grid the images used as trainnig data
        """        
        figure = plt.figure(figsize=(8, 8))

        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(self.trainset), size=(1,)).item()
            img, label = self.trainset[sample_idx]
            ax = figure.add_subplot(rows, cols, i)
            ax.set_title(self.classes[label])
            plt.axis("off")
            imshow(img)
        return plt