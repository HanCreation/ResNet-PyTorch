import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation loss and accuracy.

    Args:
        history: dict with keys 'train_loss','val_loss','train_acc','val_acc'.
        save_path: filepath to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs. Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training vs. Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix heatmap.

    Args:
        cm: confusion matrix array.
        classes: list of class names.
        save_path: filepath to save the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_random_predictions(model, test_dataset, device, num_images=6, save_path='sample_predictions.png'):
    """
    Plot random test images with true and predicted labels.

    Args:
        model: trained PyTorch model.
        test_dataset: dataset object with __getitem__ and .classes.
        device: torch.device.
        num_images: number of samples.
        save_path: filepath to save the plot.
    """
    model.eval()
    indices = random.sample(range(len(test_dataset)), num_images)
    images, true_labels, preds = [], [], []
    for idx in indices:
        img, label = test_dataset[idx]
        images.append(img)
        true_labels.append(label)
        inp = img.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            _, pred = torch.max(out, 1)
            preds.append(pred.item())

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes):
        img = images[i]
        img = img.cpu()
        img = img * 0.5 + 0.5  # unnormalize if needed
        npimg = np.transpose(img.numpy(), (1, 2, 0))
        ax.imshow(npimg)
        ax.set_title(f"True: {test_dataset.classes[true_labels[i]]}\nPred: {test_dataset.classes[preds[i]]}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show() 