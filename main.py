from data_manipulation.cifar_10 import CifarDataLoader
from models import LeNet_5, VGG_16
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from trainer.trainer import Trainer


def main():
    data = CifarDataLoader()

    #model = LeNet_5(n_inputs=3, n_classes=10)
    model = VGG_16(n_classes=10)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)


    trainer = Trainer(model, criterion, optimizer)


    trainer.train(data.trainloader)
    trainer.test(data.testloader)

    model.save_model()

if __name__ == '__main__':
    main()