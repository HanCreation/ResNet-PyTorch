"""
Examples of how to use the configuration system for different scenarios.
This file is not meant to be run directly, but to provide examples
of how to use the config system in various scenarios.
"""

import os
import sys

# Add src directory to the Python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import Config

# Example 1: Quick training on a small dataset
def example_quick_training():
    # Create a configuration for quick training
    quick_config = Config(
        model_name='resnet18',      # Use a smaller model
        batch_size=256,             # Use larger batches
        epochs=5,                   # Train for fewer epochs
        learning_rate=0.01,         # Standard learning rate
        data_dir='./data',          # Standard data directory
        use_augmentation=True       # Use data augmentation
    )
    
    # You would then use this config for training:
    # python train.py --model resnet18 --batch_size 256 --epochs 5 --data_dir ./data


# Example 2: Full training with a large model
def example_full_training():
    # Create a configuration for extensive training
    full_config = Config(
        model_name='resnet50',      # Use a larger model
        batch_size=64,              # Smaller batches (for larger model)
        epochs=100,                 # Train for many epochs
        learning_rate=0.005,        # Lower learning rate
        weight_decay=1e-4,          # More regularization
        optimizer='sgd',            # Use SGD
        lr_scheduler='multistep',   # Use multistep scheduler
        milestones=[30, 60, 90],    # Learning rate drops at these epochs
        gamma=0.1,                  # Learning rate multiplier
        use_augmentation=True,      # Use data augmentation
        use_random_erase=True,      # Use random erasing
        data_dir='./data'           # Standard data directory
    )
    
    # You would then use this config for training:
    # python train.py --model resnet50 --batch_size 64 --epochs 100 --lr 0.005 --weight_decay 1e-4 --scheduler multistep --use_random_erase


# Example 3: Fine-tuning from a pre-trained model
def example_fine_tuning():
    # Create a configuration for fine-tuning
    finetune_config = Config(
        model_name='resnet50',      # Use pre-trained model
        batch_size=32,              # Small batches for fine-tuning
        epochs=10,                  # Few epochs
        learning_rate=0.001,        # Low learning rate
        weight_decay=1e-5,          # Light regularization
        optimizer='adam',           # Use Adam optimizer
        lr_scheduler='cosine',      # Cosine annealing
        resume=True,                # Resume from checkpoint
        data_dir='./data'           # Standard data directory
    )
    
    # You would then use this config for fine-tuning:
    # python train.py --model resnet50 --batch_size 32 --epochs 10 --lr 0.001 --weight_decay 1e-5 --optimizer adam --scheduler cosine --resume


# Example 4: Evaluation only
def example_evaluation():
    # Create a configuration for evaluation
    eval_config = Config(
        model_name='resnet50',      # Model to evaluate
        batch_size=128,             # Larger batches for evaluation
        data_dir='./data',          # Standard data directory
        eval_only=True              # Only run evaluation, no training
    )
    
    # You would then use this config for evaluation:
    # python train.py --model resnet50 --batch_size 128 --eval_only


# Example 5: Using a different data directory
def example_different_data():
    # Create a configuration with a different data directory
    data_config = Config(
        model_name='resnet18',      # Standard model
        batch_size=128,             # Standard batch size
        epochs=30,                  # Standard epochs
        data_dir='/path/to/custom/data'  # Custom data directory
    )
    
    # You would then use this config:
    # python train.py --model resnet18 --batch_size 128 --epochs 30 --data_dir /path/to/custom/data


# Command line examples

"""
# Basic training with default settings:
python train.py

# Train ResNet-50 with a larger batch size:
python train.py --model resnet50 --batch_size 256

# Train for more epochs with learning rate scheduling:
python train.py --epochs 100 --scheduler multistep

# Resume training from a checkpoint:
python train.py --resume

# Use Adam optimizer instead of SGD:
python train.py --optimizer adam --lr 0.001

# Evaluate a trained model without training:
python train.py --eval_only

# Train with a different data directory:
python train.py --data_dir /path/to/custom/data

# Run inference on an image:
python inference.py --image path/to/image.jpg

# Run inference with a specific checkpoint:
python inference.py --image path/to/image.jpg --checkpoint path/to/checkpoint.pth

# Run inference with a different model architecture:
python inference.py --image path/to/image.jpg --model resnet50

# Show more or fewer top predictions:
python inference.py --image path/to/image.jpg --top_k 5
""" 