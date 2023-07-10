import torch
import numpy as np


def im_convert(tensor):  
    """
    Converts a tensor into an image
    """
    image = tensor.cpu().clone().detach().numpy() # This process will happen in normal cpu.
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)
    return image




def prepare_device():
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device
