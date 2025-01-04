"""
Provides methods to generate graphs and plots, to
describe various performance metrics.
"""

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


def roc_curve_graph(y_test, y_pred, classes: pd.Series | list,
                    folder_path: str):
    """
    Generates and saves the Receiver Operating Characteristic (ROC) curve for multi-class classification.
    This function plots the ROC curve for each class in a multi-class setting and saves the plot to the specified folder.

    Args:
        y_test (array-like): True labels of the test set, typically an array or pandas Series.
        y_pred (array-like): Predicted probabilities for each class, typically a 2D array or pandas DataFrame.
        classes (pd.Series or list): A list or pandas Series containing the class labels.
        folder_path (str): The directory path where the ROC curve plot will be saved
    """ 
    y_test_bin = label_binarize(y_test, classes=classes)

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # configure the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for 1D Convolutional Neural Network")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(folder_path + '/roc_curve.png')


def accuracy_curve(history: History, folder_path: str):
    """
    Plots and saves the training and validation accuracy curve over epochs for a given model's history.
    This function visualizes the accuracy performance during training and validation for each epoch.

    Args:
        history (History): The history object returned by the `fit()` method of a Keras model.
        
        folder_path (str): The directory path where the accuracy curve plot will be saved.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path + '/accuracy.png')
    

def confusion_matrix_graph(y_test, y_pred, classes: pd.Series | list,
                           folder_path: str):
    """
    Plots and saves the confusion matrix for a classification model's predictions.

    Args:
        y_test (array-like or pd.Series): True labels for the test set.
        y_pred (array-like or pd.Series): Predicted labels by the model for the test set.
        classes (pd.Series | list): List or series containing the names of the classes.
                                    These will be used as the labels on the axes of the confusion matrix.
        folder_path (str): Directory where the confusion matrix plot will be saved.
    """
    _, ax = plt.subplots(figsize=(12, 12))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels= classes)
    # plt.figure(figsize= (12, 12))
    ax.set_title("Confusion Matrix")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(folder_path + '/confusion_matrix.png')
    plt.close()


def plot_class_wise_accuracy(y_test, y_pred, classes: pd.Series | list, folder_path: str):
    """
    Plots the class-wise accuracy of a classification model.

    Args:
        y_test (array-like or pd.Series): True labels for the test set.
        y_pred (array-like or pd.Series): Predicted labels for the test set.
        classes (pd.Series or list): List or series containing the class labels.
                                     These will be used for the x-axis and labeling the bars.
        folder_path (str): The directory where the class-wise accuracy plot will be saved.
    """
    class_accuracies = []
    for i, _ in enumerate(classes):
        indices = np.where(np.array(y_test) == i)
        correct = np.sum(np.array(y_pred)[indices] == i)
        total = len(indices[0])
        class_accuracy = correct / total if total > 0 else 0
        class_accuracies.append(class_accuracy)
    
    plt.figure(figsize=(10, 8))
    plt.bar(classes, class_accuracies)
    plt.grid()
    plt.xlabel("Classes")
    plt.ylabel("Accuracy")
    plt.title("Class-Wise Accuracy")
    plt.ylim(0, 1)
    # plt.xticks(ha="right")
    for i, acc in enumerate(class_accuracies):
        plt.text(i, acc + 0.02, f"{acc:.2f}", ha="center", va="bottom", fontsize=10)
    
    plt.savefig(folder_path + '/class_accuracy.png')