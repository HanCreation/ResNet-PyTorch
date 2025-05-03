import argparse
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys

# Ensure project root is on sys.path to import config and src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from src directory
from src.models.ResNet.model import resnet18, resnet34, resnet50, resnet101, resnet152
from config import Config, config

def get_args():
    parser = argparse.ArgumentParser(description='ResNet inference on CIFAR-10')
    parser.add_argument('--model', type=str, default=config.model_name, 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='ResNet model architecture')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--top_k', type=int, default=3, help='Show top K predictions')
    
    return parser.parse_args()

def preprocess_image(image_path):
    """
    Preprocess the input image for model inference
    """
    # Define the same normalization as used in training
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image

def predict(model, image, device):
    """
    Perform inference on the input image
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    return predicted.item(), probabilities.cpu().numpy()

def main():
    args = get_args()
    
    # Create a configuration for inference
    inference_config = Config(
        model_name=args.model
    )
    
    # Set device
    device = inference_config.device
    print(f"Using device: {device}")
    
    # Create model based on selected architecture
    model_functions = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }
    
    model = model_functions[inference_config.model_name](num_classes=inference_config.num_classes)
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Use default checkpoint path from config
        checkpoint_path = os.path.join(inference_config.save_dir, f'{inference_config.model_name}_cifar10.pth')
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    model = model.to(device)
    
    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Preprocess image
    image = preprocess_image(args.image)
    
    # Predict
    predicted_class, probabilities = predict(model, image, device)
    
    # Print results
    print(f"\nPredicted class: {class_names[predicted_class]}")
    print("\nClass probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{class_names[i]}: {prob*100:.2f}%")
    
    # Get top-k predictions
    top_k = min(args.top_k, len(class_names))
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    print(f"\nTop {top_k} predictions:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {class_names[idx]}: {probabilities[idx]*100:.2f}%")

if __name__ == '__main__':
    main() 