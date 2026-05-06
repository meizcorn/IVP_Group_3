import torch
import matplotlib.pyplot as plt

from config import OUTPUT_DIR
from evaluate import plot_confusion_matrix


HOG_ACC = 0.9950
CNN_ACC = 0.9938


def plot_accuracy_comparison():
    models = ["HOG-SVM", "CNN"]
    accuracies = [HOG_ACC, CNN_ACC]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, accuracies)

    plt.ylim(0.99, 1.0)
    plt.ylabel("Validation Accuracy")
    plt.title("HOG-SVM vs CNN Validation Accuracy")

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            acc,
            f"{acc:.4f}",
            ha="center",
            va="bottom"
        )

    save_path = OUTPUT_DIR / "model_accuracy_comparison.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Saved: {save_path}")


def plot_cnn_confusion_matrix():
    data = torch.load("models/predictions.pth", weights_only=False)

    y_true = data["y_true"]
    y_pred = data["y_pred"]

    save_path = OUTPUT_DIR / "confusion_matrix_cnn.png"
    plot_confusion_matrix(y_true, y_pred, save_path=save_path)


def main():
    plot_accuracy_comparison()
    plot_cnn_confusion_matrix()


if __name__ == "__main__":
    main()