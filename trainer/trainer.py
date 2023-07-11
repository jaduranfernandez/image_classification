import torch
from utils import prepare_device
import numpy as np
#from trainer import MetricsTracker


class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = prepare_device()
        self.model = model.to(self.device)

        

    def train(self, dataloader, metricsTracker):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        train_loss, correct = 0, 0
        
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)    #   Makes a prediction given a specified data(X)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 1000 == 0:
                current_train_loss, current = loss.item(), batch * len(X)                
                print(f"loss: {current_train_loss:>7f}  [{current:>5d}/{size:>5d}]")
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                train_loss += current_train_loss
        train_loss /= num_batches
        correct /= size
        metricsTracker.add_train_metrics(train_loss, correct)


    def test(self, dataloader, metricsTracker):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0        
        with torch.no_grad():
            for X, y in dataloader:                
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
                pred_value = pred.argmax(1) 
                test_loss += self.criterion(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                metricsTracker.add_prediction(y.to('cpu'), pred_value.to('cpu'))

        test_loss /= num_batches
        correct /= size
        metricsTracker.add_test_metrics(test_loss, correct)
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

