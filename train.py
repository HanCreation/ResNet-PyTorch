import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.ResNet.model import resnet18, resnet34, resnet50, resnet101, resnet152
from src.data.dataset import CIFAR10Data
from src.utils.trainer import Trainer
from src.visualization.visualize import plot_training_history, plot_confusion_matrix, plot_random_predictions
from config import Config, config

def get_args():
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10')
    # Model configuration
    parser.add_argument('--model', type=str, default=config.model_name, 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='ResNet model architecture')
    
    # Data settings
    parser.add_argument('--data_dir', type=str, default=config.data_dir, help='Data directory')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=config.num_workers, help='Number of worker threads for data loading')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=config.epochs, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=config.learning_rate, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=config.momentum, help='Momentum for SGD optimizer')
    parser.add_argument('--optimizer', type=str, default=config.optimizer, choices=['sgd', 'adam'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default=config.lr_scheduler, 
                        choices=['cosine', 'step', 'multistep', 'none'], help='Learning rate scheduler')
    
    # Paths
    parser.add_argument('--save_dir', type=str, default=config.save_dir, help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default=config.log_dir, help='Directory to save logs')
    
    # Training options
    parser.add_argument('--resume', action='store_true', default=config.resume, help='Resume training from checkpoint')
    parser.add_argument('--seed', type=int, default=config.seed, help='Random seed')
    parser.add_argument('--eval_only', action='store_true', default=config.eval_only, help='Only evaluate model')
    
    # Data augmentation
    parser.add_argument('--use_augmentation', action='store_true', default=config.use_augmentation, help='Use data augmentation')
    parser.add_argument('--use_random_erase', action='store_true', default=config.use_random_erase, help='Use random erasing')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Create config from args
    custom_config = Config(
        model_name=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        optimizer=args.optimizer,
        lr_scheduler=args.scheduler,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume=args.resume,
        seed=args.seed,
        eval_only=args.eval_only,
        use_augmentation=args.use_augmentation,
        use_random_erase=args.use_random_erase
    )
    
    # Print configuration
    print(custom_config)
    
    # Set random seeds for reproducibility
    torch.manual_seed(custom_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(custom_config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set device
    device = custom_config.device
    print(f"Using device: {device}")
    
    # Prepare data
    data = CIFAR10Data(custom_config)
    train_loader = data.get_train_dataloader()
    val_loader = data.get_val_dataloader()
    test_loader = data.get_test_dataloader()
    
    # Create model based on selected architecture
    model_functions = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }
    
    model = model_functions[custom_config.model_name](num_classes=custom_config.num_classes)
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    if custom_config.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=custom_config.learning_rate,
            momentum=custom_config.momentum,
            weight_decay=custom_config.weight_decay
        )
    elif custom_config.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=custom_config.learning_rate,
            weight_decay=custom_config.weight_decay
        )
    
    # Define learning rate scheduler
    if custom_config.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=custom_config.epochs)
    elif custom_config.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=custom_config.step_size, gamma=custom_config.gamma)
    elif custom_config.lr_scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=custom_config.milestones, gamma=custom_config.gamma)
    else:
        scheduler = None
    
    # Checkpoint path
    checkpoint_path = os.path.join(custom_config.save_dir, f'{custom_config.model_name}_cifar10.pth')
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if custom_config.resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Create trainer
    trainer = Trainer(model, optimizer, criterion, device)
    
    # Only evaluate if requested
    if custom_config.eval_only:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path} for evaluation")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print("\nEvaluating on test set...")
            test_acc, report, conf_matrix, _, _ = trainer.test(test_loader, data.classes)
            print(f"Test Accuracy: {test_acc:.2f}%")
            print("\nClassification Report:")
            print(report)
            
            # Plot confusion matrix
            plot_confusion_matrix(conf_matrix, data.classes)
            
            # Plot sample predictions
            plot_random_predictions(model, data.test_dataset, device)
            return
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=custom_config.epochs - start_epoch,
        scheduler=scheduler,
        save_path=checkpoint_path
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_acc, report, conf_matrix, _, _ = trainer.test(test_loader, data.classes)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, data.classes)
    
    # Plot sample predictions
    print("\nGenerating sample predictions...")
    plot_random_predictions(model, data.test_dataset, device)

if __name__ == '__main__':
    main() 