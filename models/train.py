import sys
sys.path.append('../')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import numpy as np
from tqdm import tqdm

from random import seed
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import process_data, process_data_2, accuracy, process_data_batched


################################################################################################################################################################################################################
####################################################################### Model's Fit Function ###################################################################################################################
################################################################################################################################################################################################################

def validate(model, val_dl, loss_func, device, dataset_name, use_batched=False):
    """
    Validate model on validation set.
    
    Args:
        model: the neural network model
        val_dl: validation dataloader
        loss_func: loss function to use
        device: torch device
        dataset_name: name of dataset
        use_batched: whether to use batched processing
    
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    with torch.no_grad():
        # Process validation data
        if use_batched:
            val_images, val_texts, val_labels = process_data_batched(val_dl, dataset_name, device)
        else:
            val_images, val_texts, val_labels = process_data(val_dl, dataset_name)
            val_images = val_images.to(device)
            val_texts = val_texts.to(device)
            val_labels = val_labels.to(device)
        
        val_labels = val_labels.long()
        
        # Process in batches to avoid memory issues
        batch_size = 32  # Smaller batch for validation
        n_samples = len(val_labels)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_images = val_images[start_idx:end_idx]
            batch_texts = val_texts[start_idx:end_idx]
            batch_labels = val_labels[start_idx:end_idx]
            
            # Forward pass
            outputs = model(batch_images, batch_texts.to(torch.float32), "complete")
            loss = loss_func(outputs, batch_labels)
            
            total_loss += loss.item()
            total_acc += accuracy(outputs, batch_labels)
            n_batches += 1
    
    return total_loss / n_batches, total_acc / n_batches

def fit(epochs, model, train_dl, val_dl, optimizer, lr_scheduler, batch_num, 
        dataset_name, model_config, class_weights=None, device=None, use_batched=False):
    """
    Train the model with validation and early stopping.
    
    Args:
        epochs: number of training epochs
        model: the neural network model
        train_dl: training dataloader
        val_dl: validation dataloader (can be None for backward compatibility)
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler
        batch_num: batch size
        dataset_name: name of dataset ('padufes20' or other)
        model_config: model configuration string
        class_weights: weights for each class to handle imbalance
        device: torch device (if None, will get from model)
        use_batched: whether to use memory-efficient batched processing
    """
    opt = optimizer
    sched = lr_scheduler
    
    # Setup loss function with class weights if provided
    if class_weights is not None:
        print(f"Using weighted cross entropy with weights: {class_weights}")
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_func = nn.CrossEntropyLoss()
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Create checkpoint directory
    path = "sample_data/checkpoints/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get device from model if not provided
    if device is None:
        device = next(model.parameters()).device
    
    print(f"Training on device: {device}")
    print("Calculating features for training set...")
    
    # Process training data with error handling
    try:
        if use_batched:
            image_input, text_input, label = process_data_batched(train_dl, dataset_name, device)
        else:
            image_input, text_input, label = process_data(train_dl, dataset_name)
    # Move tensors to device
            image_input = image_input.to(device)
            text_input = text_input.to(device)
            label = label.to(device)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU OOM during data processing! Switching to batched processing...")
            torch.cuda.empty_cache()
            image_input, text_input, label = process_data_batched(train_dl, dataset_name, device, batch_size=16)
        else:
            raise e
    
    print(f"Feature sizes: ViT({image_input.size()}); Text({text_input.size()}); Labels({label.size()})")
    
    # Print class distribution
    unique_labels, counts = torch.unique(label, return_counts=True)
    print("\nClass distribution in training:")
    for l, c in zip(unique_labels, counts):
        print(f"  Class {l}: {c} samples ({c/len(label)*100:.1f}%)")
    
    print("\nStarting training...")
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
       
        # Training metrics for this epoch
        epoch_loss = 0
        epoch_acc = 0
        
        label = label.long()
        n_batches = int(len(label) / batch_num)
        
        # Progress bar for batches
        pbar = tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx in pbar:
            # Random sampling for each batch
            batch_indices = []
            seed(time.perf_counter())
            for _ in range(batch_num):
                batch_indices.append(random.randint(0, len(label)-1))
            
            # Get batch data
            batch_images = image_input[batch_indices]
            batch_texts = text_input[batch_indices]
            batch_labels = label[batch_indices]
            
            # Ensure data is on correct device
            batch_images = batch_images.to(device)
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            try:
                preds = model(batch_images, batch_texts.to(torch.float32), model_config)
                loss = loss_func(preds, batch_labels)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nGPU OOM at batch {batch_idx}. Skipping batch...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            opt.zero_grad()

            # Update metrics
            batch_acc = accuracy(preds, batch_labels)
            epoch_loss += loss.item()
            epoch_acc += batch_acc
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.4f}'
            })
        
        # Calculate epoch averages
        avg_train_loss = epoch_loss / n_batches
        avg_train_acc = epoch_acc / n_batches
        
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        # Validation phase
        if val_dl is not None:
            val_loss, val_acc = validate(model, val_dl, loss_func, device, dataset_name, use_batched)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f'\nEpoch {epoch+1}/{epochs}:')
            print(f'  Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}')
            print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
            
            # Learning rate scheduler step (based on validation loss)
            sched.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_path = os.path.join(path, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        else:
            # No validation set - use training loss for scheduler
            sched.step(avg_train_loss)
            print(f'\nEpoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}')
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'  Learning rate: {current_lr:.2e}')
    
    # Save final model
    final_model_path = os.path.join(path, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    torch.save(history, os.path.join(path, "training_history.pt"))
    
    # Save losses and accuracies to text files in checkpoints folder
    # These provide easy access without needing to load PyTorch files
    with open(os.path.join(path, "loss_training.txt"), "w") as f:
        f.write('\n'.join(str(loss) for loss in train_losses))
    
    with open(os.path.join(path, "acc_training.txt"), "w") as f:
        f.write('\n'.join(str(acc.item() if hasattr(acc, 'item') else acc) for acc in train_accs))
    
    # Also save validation metrics if available
    if val_losses:
        with open(os.path.join(path, "loss_validation.txt"), "w") as f:
            f.write('\n'.join(str(loss) for loss in val_losses))

        with open(os.path.join(path, "acc_validation.txt"), "w") as f:
            f.write('\n'.join(str(acc.item() if hasattr(acc, 'item') else acc) for acc in val_accs))
    
    print("\nTraining completed!")
