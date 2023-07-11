import numpy as np
from utils import plot_confusion_mat
import matplotlib.pyplot as plt


class MetricsTracker():
    """
    Stores performance data such as:
        - Loss (training & validation)
        - Accuracy (training & validation)
        - Confusion matrix
    """
    def __init__(self, n_classes, n_epochs):
        self.loss_train = np.zeros(n_epochs)
        self.loss_test = np.zeros(n_epochs)
        self.accuracy_train = np.zeros(n_epochs)
        self.accuracy_test = np.zeros(n_epochs)
        self.confusion_mat = np.zeros([n_classes, n_classes])
        self.n_epochs = n_epochs
        self.current_epoch = 1

    def add_train_metrics(self, avg_loss, avg_accuracy):
        self.loss_train[self.current_epoch-1] = avg_loss
        self.accuracy_train[self.current_epoch-1] = avg_accuracy


    def add_test_metrics(self, avg_loss, avg_accuracy):
        self.loss_test[self.current_epoch-1] = avg_loss
        self.accuracy_test[self.current_epoch-1] = avg_accuracy


    def add_prediction(self, true_value, pred_value):
        self.confusion_mat[true_value,pred_value] += 1
        self.confusion_mat[pred_value,true_value] += 1


    def clean_confussion_mat(self):
        n_classes = self.confusion_mat.shape[0]
        self.confusion_mat = np.zeros([n_classes, n_classes])


    def update(self):
        if(self.current_epoch < self.n_epochs):
            self.current_epoch += 1


    def plot_results(self, labels = None, save: bool = False):
        normalized_matrix = self.confusion_mat.astype('float') / self.confusion_mat.sum(axis=1)[:, np.newaxis]
        plot_confusion_mat(normalized_matrix, labels = labels)
        plt.show()    