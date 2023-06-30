from data_manipulation.data_loader import DataLoader
from model.base_model import NeuralNetwork
from model.trainer import Trainer
import torch

data = DataLoader()
#data.show_subplot(4,4)

model = NeuralNetwork()
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

trainer = Trainer(model, criterion, optimizer)


epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainer.train(data.trainloader)
    trainer.test(data.testloader)
print("Done!")
