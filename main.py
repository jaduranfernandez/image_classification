from data_manipulation.data_loaders import CifarDataLoader
from models.LeNet_5 import LeNet_5
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from models.trainer import Trainer


model = LeNet_5(n_inputs=3, n_classes=10)
print(model)
data = CifarDataLoader()


criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)


model.train()
trainer = Trainer(model, criterion, optimizer)
#print(data.valid_sampler)

trainer.train(data.trainloader)
