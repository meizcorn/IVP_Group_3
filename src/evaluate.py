import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

##some results for accuracy
def print_classification_results(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    print("\nValidation Accuracy:")
    print(f"{accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

##confusion matrix
def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(range(10))
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to: {save_path}")

    plt.show()