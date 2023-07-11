from data_manipulation.cifar_10 import CifarDataLoader
from models import LeNet_5, VGG_16, SimpleNet
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from trainer.trainer import Trainer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils import make_confusion_matrix



def main():
    data = CifarDataLoader(batch_size = 16, num_workers = 2)

    #model = LeNet_5(n_inputs=3, n_classes=10)
    #model = VGG_16(n_classes=10)
    model = SimpleNet(n_classes=10)
    criterion = CrossEntropyLoss()
    #optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = Adam(model.parameters(), lr = 0.001)
    epochs = 10


    trainer = Trainer(model, criterion, optimizer)

    print(data.nClasses)

    for epoch in range(1,epochs+1):
        print("Epoch {} \n--------------------".format(epoch))
        trainer.train(data.trainloader)
        true_values, pred_values = trainer.test(data.testloader, data.nClasses)

    # Calcular la matriz de confusión
    cm = confusion_matrix(true_values, pred_values)
    make_confusion_matrix(cm, categories=data.classes)

    model.save_model()

if __name__ == '__main__':
    main()