import os
from datetime import datetime

class Config:
    """
    Configuration settings for the ResNet CIFAR-10 project.
    This class centralizes all hyperparameters and settings.
    """
    
    # Data settings
    data_dir = './data'
    batch_size = 128
    num_workers = 4
    
    # Model settings
    model_name = 'resnet18'  # One of: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    num_classes = 10  # CIFAR-10 has 10 classes
    
    # Training hyperparameters
    epochs = 30
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    lr_scheduler = 'cosine'  # Options: 'cosine', 'step', 'multistep', 'none'
    
    # Step LR scheduler settings
    step_size = 10
    gamma = 0.1
    
    # MultiStep LR scheduler settings
    milestones = [15, 25]
    
    # Optimizer settings
    optimizer = 'sgd'  # Options: 'sgd', 'adam'
    
    # Paths
    save_dir = './checkpoints'
    log_dir = './logs'
    
    # Random seed for reproducibility
    seed = 42
    
    # Data augmentation settings
    use_augmentation = True
    random_crop_padding = 4
    use_random_flip = True
    use_random_erase = False
    random_erase_prob = 0.5
    
    # Device settings - will be auto-detected
    use_cuda = True
    device = None  # Will be set at runtime
    
    # Resume training
    resume = False
    checkpoint_path = None  # Will be set based on model_name if resume is True
    
    # Evaluation settings
    eval_only = False
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with custom settings.
        Override default values with provided arguments.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Create a new run folder inside the log_dir for each training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self.log_dir, timestamp)
        self.save_dir = self.log_dir
        
        # Create directories if they don't exist
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set device
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        
        # Set checkpoint path if resuming
        if self.resume and not self.checkpoint_path:
            self.checkpoint_path = os.path.join(self.save_dir, f'{self.model_name}_cifar10.pth')
    
    def __str__(self):
        """Return a string representation of the configuration"""
        config_str = "Configuration Settings:\n"
        for attr, value in vars(self).items():
            if not attr.startswith('__'):
                config_str += f"  {attr}: {value}\n"
        return config_str
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {key: value for key, value in vars(self).items() 
                if not key.startswith('__') and not callable(value)}


# Create a default configuration
config = Config() 