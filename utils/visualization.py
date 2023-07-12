import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_mat(confusion_mat: np.ndarray, labels: list[str] = None):
    """
    Plots a confusion matrix
    """
    mat_dim = confusion_mat.shape[0]
    if(not(labels) or len(labels) != mat_dim):
        labels = [str(i) for i in range(mat_dim)]

    df_cm = pd.DataFrame(confusion_mat, index = [i for i in labels],
                  columns = [i for i in labels])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    return plt


def plot_lines(n_epochs: int, y_coord_train: np.array, y_coord_test: np.array, title: str):
    epochs = np.arange(1,n_epochs+1)
    plt.plot(epochs, y_coord_train, '-o', label = "Train")
    plt.plot(epochs, y_coord_test, '-o',label = "Test")
    plt.xticks(epochs)
    plt.title(title)
    plt.legend()
    return plt
