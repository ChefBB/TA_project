#########################
# roc curve
#########################
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


#########################
# ROC Curve
#########################
def roc_curve_graph(y_test, y_pred, classes: pd.Series | list,
                    folder_path: str):
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


#########################
# Training and Validation Loss/Accuracy Curve
#########################
def accuracy_curve(history: History, folder_path: str):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder_path + '/accuracy.png')
    

#########################
# Confusion matrix graph
#########################
def confusion_matrix_graph(y_test, y_pred, classes: pd.Series | list,
                           folder_path: str):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels= classes)
    plt.figure(figsize= (12, 12))
    plt.title("Confusion Matrix")
    plt.xticks(rotation= 90)
    plt.savefig(folder_path + '/confusion_matrix.png')


#########################
# Confusion matrix graph
#########################
def plot_class_wise_accuracy(y_test, y_pred, classes: pd.Series | list, folder_path: str):
    class_accuracies = []
    for i, label in enumerate(classes):
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