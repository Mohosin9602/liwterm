import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.metrics import classification_report, confusion_matrix

from torchvision import transforms
from torch.utils.data import DataLoader
from utils import process_data, process_data_2, accuracy, process_data_batched

################################################################################################################################################################################################################
####################################################################### Confusion Matrix #######################################################################################################################
################################################################################################################################################################################################################

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    """
    Plot and save confusion matrix.
    
    Args:
        df_confusion: pandas DataFrame with confusion matrix
        title: title for the plot
        cmap: colormap to use
    """
    plt.figure(figsize=(10, 8))
    plt.matshow(df_confusion, cmap=cmap, fignum=1)
    plt.colorbar()
    
    # Get the actual number of classes from the confusion matrix
    n_classes = len(df_confusion.columns)
    tick_marks = np.arange(n_classes)
    
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    
    # Add value annotations
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(df_confusion.iloc[i, j]), 
                    ha='center', va='center', color='white' if df_confusion.iloc[i, j] > df_confusion.max().max()/2 else 'black')
    
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    
    # Save figure instead of showing (for remote environments)
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=150)
    plt.close()  # Close to prevent memory leak
    print("Confusion matrix saved to confusion_matrix.png")

################################################################################################################################################################################################################
####################################################################### Model's Test Functions #################################################################################################################
################################################################################################################################################################################################################
    
def test_partial(model, test_data, batch_num, model_config, dataset_name="padufes20", use_batched=False):
    """
    Test model performance with detailed metrics.
    
    Args:
        model: trained model
        test_data: test dataloader
        batch_num: batch size
        model_config: model configuration
        dataset_name: name of dataset
        use_batched: whether to use memory-efficient batched processing
    """
    # Class names mapping for better readability
    CLASS_NAMES = {
        0: "NEV",  # Nevus
        1: "BCC",  # Basal Cell Carcinoma
        2: "ACK",  # Actinic Keratosis
        3: "SEK",  # Seborrheic Keratosis
        4: "SCC",  # Squamous Cell Carcinoma
        5: "MEL",  # Melanoma
        6: "UNK6", # Unknown class 6 (if exists)
        7: "UNK7"  # Unknown class 7 (if exists)
    }
    
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = nn.CrossEntropyLoss()
    out_labels = []
    out_preds = []
    out_probs = []
    
    # Get device from model
    device = next(model.parameters()).device
    
    print(f"Testing on device: {device}")
    print("Processing test data...")
    
    with torch.no_grad():
        # Process test data
        try:
            if use_batched:
                image_input, text_input, target = process_data_batched(test_data, dataset_name, device)
            else:
                image_input, text_input, target = process_data(test_data, dataset_name)
                # Move tensors to device
                image_input = image_input.to(device)
                text_input = text_input.to(device)
                target = target.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU OOM during test data processing! Switching to batched processing...")
                torch.cuda.empty_cache()
                image_input, text_input, target = process_data_batched(test_data, dataset_name, device, batch_size=16)
            else:
                raise e
        
        overall_acc = 0
        n_batches = int(len(target) / batch_num)
        print(f"\nNumber of test batches: {n_batches}")
        print(f"Total test samples: {len(target)}")
        
        target = target.long()
        batch_start = 0
        
        # Process test data in batches
        for batch_idx in range(n_batches):
            # Get batch indices
            batch_end = min(batch_start + batch_num, len(target))
            batch_indices = range(batch_start, batch_end)
            
            # Get batch data
            batch_images = image_input[batch_indices]
            batch_texts = text_input[batch_indices]
            batch_labels = target[batch_indices]
            
            # Ensure data is on correct device
            batch_images = batch_images.to(device)
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass with timing
            start_time = time.time()
            try:
                output = model(batch_images, batch_texts.to(torch.float32), model_config)
                batch_time = time.time() - start_time
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU OOM at test batch {batch_idx}. Skipping...")
                    torch.cuda.empty_cache()
                    batch_start += batch_num
                    continue
                else:
                    raise e
            
            # Calculate loss and accuracy
            test_loss = loss_func(output, batch_labels)
            acc = accuracy(output, batch_labels)
            overall_acc += float(acc)
            
            # Get predictions and probabilities
            probs = F.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            
            # Store results
            out_preds.extend(preds.cpu().numpy())
            out_labels.extend(batch_labels.cpu().numpy())
            out_probs.extend(probs.cpu().numpy())
            
            # Print batch results
            print(f"\nBatch {batch_idx + 1}/{n_batches}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  Inference time: {batch_time:.3f}s ({batch_time/len(batch_indices)*1000:.1f}ms per sample)")
            
            batch_start += batch_num
        
        # Handle remaining samples
        if batch_start < len(target):
            remaining_indices = range(batch_start, len(target))
            batch_images = image_input[remaining_indices]
            batch_texts = text_input[remaining_indices]
            batch_labels = target[remaining_indices]
            
            output = model(batch_images, batch_texts.to(torch.float32), model_config)
            probs = F.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            
            out_preds.extend(preds.cpu().numpy())
            out_labels.extend(batch_labels.cpu().numpy())
            out_probs.extend(probs.cpu().numpy())
        
        # Calculate overall metrics
        overall_acc /= n_batches
        
        # Convert lists to numpy arrays
        out_labels = np.array(out_labels)
        out_preds = np.array(out_preds)
        out_probs = np.array(out_probs)
        
        print(f"\n{'='*60}")
        print("OVERALL TEST RESULTS")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
        print(f"Total correct: {np.sum(out_labels == out_preds)}/{len(out_labels)}")
        
        # Get unique classes in test set
        unique_classes = np.unique(out_labels)
        class_names = [CLASS_NAMES.get(c, f"Class_{c}") for c in unique_classes]
        
        # Detailed classification report
        print("\nClassification Report:")
        print("-" * 60)
        report = classification_report(out_labels, out_preds, 
                                     labels=unique_classes,
                                     target_names=class_names,
                                     digits=4)
        print(report)
        
        # Save classification report
        with open("classification_report.txt", "w") as f:
            f.write(f"Overall Accuracy: {overall_acc:.4f}\n")
            f.write(f"Total correct: {np.sum(out_labels == out_preds)}/{len(out_labels)}\n\n")
            f.write(report)
        
        # Per-class accuracy
        print("\nPer-Class Accuracy:")
        print("-" * 40)
        for class_idx in unique_classes:
            class_mask = out_labels == class_idx
            class_acc = np.mean(out_preds[class_mask] == class_idx)
            class_name = CLASS_NAMES.get(class_idx, f"Class_{class_idx}")
            print(f"{class_name:>10}: {class_acc:.4f} ({np.sum(class_mask)} samples)")
        
        # Confusion matrix
        cm = confusion_matrix(out_labels, out_preds)
        df_confusion = pd.DataFrame(cm, index=class_names, columns=class_names)
        
        print("\nConfusion Matrix:")
        print(df_confusion)
        
        # Save confusion matrix as CSV
        df_confusion.to_csv("confusion_matrix.csv")
        
        # Plot confusion matrix
        try:
            plot_confusion_matrix(df_confusion)
        except Exception as e:
            print(f"Warning: Could not plot confusion matrix: {e}")
        
        # Save predictions
        results_df = pd.DataFrame({
            'true_label': out_labels,
            'predicted_label': out_preds,
            'true_class': [CLASS_NAMES.get(l, f"Class_{l}") for l in out_labels],
            'predicted_class': [CLASS_NAMES.get(p, f"Class_{p}") for p in out_preds],
            'correct': out_labels == out_preds
        })
        
        # Add probability columns
        for i in range(out_probs.shape[1]):
            if i in unique_classes:
                class_name = CLASS_NAMES.get(i, f"Class_{i}")
                results_df[f'prob_{class_name}'] = out_probs[:, i]
        
        results_df.to_csv("test_predictions.csv", index=False)
        print("\nTest predictions saved to test_predictions.csv")
        
        # Analyze misclassifications
        print(f"\n{'='*60}")
        print("MISCLASSIFICATION ANALYSIS")
        print(f"{'='*60}")
        
        misclassified = results_df[~results_df['correct']]
        print(f"Total misclassified: {len(misclassified)} ({len(misclassified)/len(results_df)*100:.2f}%)")
        
        if len(misclassified) > 0:
            print("\nMost common misclassifications:")
            misclass_pairs = misclassified.groupby(['true_class', 'predicted_class']).size().sort_values(ascending=False).head(10)
            for (true_class, pred_class), count in misclass_pairs.items():
                print(f"  {true_class} → {pred_class}: {count} times")


def test(model, test_dl):
    """
    Legacy test function for backward compatibility.
    """
    print("Using legacy test function. Consider using test_partial for more detailed metrics.")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        image_input, text_input, target = process_data(test_dl)
        target = target.long()
        
        output = model(image_input, text_input['input_ids'], text_input['attention_mask'])
        
        acc = accuracy(output, target)
        print(f'\nTest set: Accuracy: {acc:.4f} ({acc*100:.2f}%)\n')

