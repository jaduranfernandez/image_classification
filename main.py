from data_manipulation.cifar_10 import CifarDataLoader
from models import LeNet_5, VGG_16, SimpleNet
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from trainer import Trainer, MetricsTracker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils import make_confusion_matrix



def main():
    data = CifarDataLoader(batch_size = 16, num_workers = 2)

    model = SimpleNet(n_classes=10)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = Adam(model.parameters(), lr = 0.001)
    epochs = 1
    metricsTracker = MetricsTracker(data.nClasses, epochs)

    trainer = Trainer(model, criterion, optimizer)


    for epoch in range(1,epochs+1):
        is_last_epoch = (epoch == epochs)
        print("Epoch {} \n--------------------".format(epoch))
        trainer.train(data.trainloader, metricsTracker)
        trainer.test(data.testloader, metricsTracker)
        if(not(is_last_epoch)):
            metricsTracker.clean_confussion_mat()
        metricsTracker.update()
    print(metricsTracker.accuracy_train)
    print(metricsTracker.accuracy_test)
    print(metricsTracker.loss_train)
    print(metricsTracker.loss_test)
    metricsTracker.plot_results(labels = data.classes)


    #model.save_model()

if __name__ == '__main__':
    main()