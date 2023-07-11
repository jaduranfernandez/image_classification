import torch
from utils import prepare_device
import numpy as np

class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = prepare_device()
        self.model = model.to(self.device)
        # fix random seeds for reproducibility
        SEED = 123
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        

    def train(self, dataloader):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)    #   Makes a prediction given a specified data(X)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 1000 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader, n_classes):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        true_values = np.zeros(n_classes)
        pred_values = np.zeros(n_classes)
        with torch.no_grad():
            for X, y in dataloader:
                true_values[y] += 1
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
                #pred_values[pred.argmax(1)] += 1
                test_loss += self.criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return true_values, pred_values

