import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_training_history(history):
    """
    Plot training and validation loss and accuracy
    
    Args:
        history: dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(conf_matrix, classes):
    """
    Plot confusion matrix as a heatmap
    
    Args:
        conf_matrix: confusion matrix from sklearn
        classes: list of class names
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_random_predictions(model, test_data, device, num_images=6):
    """
    Plot random images with their true and predicted labels
    
    Args:
        model: trained PyTorch model
        test_data: test dataset
        device: device to run inference on
        num_images: number of random images to display
    """
    model.eval()
    indices = random.sample(range(len(test_data)), num_images)
    images, labels, preds = [], [], []
    
    for idx in indices:
        img, label = test_data[idx]
        images.append(img)
        labels.append(label)
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img)
            _, pred = torch.max(output, 1)
            preds.append(pred.item())
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        ax = axes[i]
        img = images[i] / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(f'True: {test_data.dataset.classes[labels[i]]}\nPred: {test_data.dataset.classes[preds[i]]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

def visualize_feature_maps(model, img, layer_name, num_features=8):
    """
    Visualize feature maps of a specific layer
    
    Args:
        model: trained PyTorch model
        img: input image tensor (1, C, H, W)
        layer_name: name of the layer to visualize
        num_features: number of feature maps to display
    """
    # Register hook to get feature maps
    feature_maps = {}
    
    def hook_fn(module, input, output):
        feature_maps[layer_name] = output
    
    # Find the layer by name and register hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(img)
    
    # Get feature maps
    feature_map = feature_maps[layer_name][0].cpu()
    
    # Plot feature maps
    num_features = min(num_features, feature_map.size(0))
    fig, axes = plt.subplots(1, num_features, figsize=(15, 5))
    
    for i in range(num_features):
        ax = axes[i]
        ax.imshow(feature_map[i].numpy(), cmap='viridis')
        ax.set_title(f'Feature Map {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'feature_maps_{layer_name}.png')
    plt.show() 