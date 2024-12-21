#########################
# roc curve
#########################
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
import pandas as pd


#########################
# ROC Curve
#########################
def roc_curve_graph(model: Model, classes: pd.Series | list, folder_path: str, x_test, y_test):
    y_test_bin = label_binarize(y_test, classes=classes)

    y_score = model.predict(
        x_test
    )

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
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