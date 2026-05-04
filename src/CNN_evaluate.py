import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize=True):

    #convert to numpy array
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    #check if the lists are empty
    if len(y_true) == 0 or len(y_pred) == 0:
        print("ERROR: Empty y_true or y_pred")
        return
    #check if the length of y_true and y_pred match
    if len(y_true) != len(y_pred):
        print(f"ERROR: Length mismatch: {len(y_true)} vs {len(y_pred)}")
        return

    print(f"Plotting confusion matrix with {len(y_true)} samples")

    #get all labels (0-9)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    #create confusion matrix (predictions vs true labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    #normalize for easy visualization
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    #plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)

    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d")

    plt.title("Confusion Matrix")
    plt.tight_layout()

    #save confusion matrix
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.show()