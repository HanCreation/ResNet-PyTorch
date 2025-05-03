import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import RandomErasing

class CIFAR10Data:
    """
    Class to handle CIFAR10 dataset loading, preprocessing, and data loaders.
    """
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        
        # Define transforms based on config settings
        transform_list = []
        
        # Training transforms
        if config.use_augmentation:
            if config.random_crop_padding > 0:
                transform_list.append(transforms.RandomCrop(32, padding=config.random_crop_padding))
            if config.use_random_flip:
                transform_list.append(transforms.RandomHorizontalFlip())
        
        # Common transforms
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # Add random erasing if enabled
        if config.use_augmentation and config.use_random_erase:
            transform_list.append(RandomErasing(p=config.random_erase_prob))
        
        self.train_transform = transforms.Compose(transform_list)
        
        # Test transform (no augmentation)
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # Load datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.prepare_data()
        
    def prepare_data(self):
        """Load datasets and create train/val split"""
        # Download and load training data
        full_train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # Split into train and validation sets (90% - 10%)
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        # Download and load test data
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
    
    def get_train_dataloader(self, shuffle=True):
        """Return the training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_dataloader(self):
        """Return the validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_dataloader(self):
        """Return the test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    @property
    def classes(self):
        """Return the class names"""
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck'] 