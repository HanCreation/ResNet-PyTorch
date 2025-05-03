import torch
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Trainer:
    """
    Class to handle model training and evaluation
    """
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_one_epoch(self, train_loader):
        """
        Train the model for one epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc="Training")):
            # Move data to device
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update running loss
            running_loss += loss.item()
            
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Store history
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, val_loader):
        """
        Evaluate the model on validation data
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(val_loader, desc="Validating")):
                # Move data to device
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update running loss
                running_loss += loss.item()
                
        # Calculate epoch metrics
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        # Store history
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def test(self, test_loader, classes):
        """
        Test the model on test data and generate detailed metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Testing"):
                # Move data to device
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                
                # Collect predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        # Calculate accuracy
        test_acc = 100. * accuracy_score(all_labels, all_preds)
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=classes)
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return test_acc, report, conf_matrix, all_preds, all_labels
    
    def train(self, train_loader, val_loader, num_epochs, scheduler=None, save_path=None):
        """
        Train the model for multiple epochs
        """
        best_val_acc = 0.0
        
        print(f"Starting training on {self.device}...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train one epoch
            train_loss, train_acc = self.train_one_epoch(train_loader)
            
            # Evaluate
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Step scheduler if provided
            if scheduler is not None:
                scheduler.step()
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if save_path and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, save_path)
                print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes")
        
        # Return training history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accuracies,
            'val_acc': self.val_accuracies
        }
        
        return history 