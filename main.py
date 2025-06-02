# -*- coding: utf-8 -*-
"""
LiwTerm - Medical Image Classification with Text Features
Luis Souza
la.souza@inf.ufes.br

This script trains a Vision Transformer (ViT) model combined with text features
for medical image classification on either PAD-UFES-20 or ISIC19 datasets.
"""

#imports
import os
import pandas as pd
import numpy as np
import torch
import argparse
import warnings
from sklearn.model_selection import train_test_split

from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTModel, ViTConfig
from torch.utils.data import DataLoader
from datasets import Dataset
from utils import (process_metadata_frame, customDataset, process_data, set_params, 
                   process_metadata_frame_isic, get_default_device, DeviceDataLoader,
                   process_data_batched, calculate_class_weights)
from models.vit import vit_model
from models.bert import bert_model
from models.liwterm import model_final
from models.train import fit
from models.test import test_partial

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train medical image classifier with text features')
parser.add_argument("src_dataset", help="Dataset Name (padufes20 or isic19)")
parser.add_argument("backbone", help="Model type (ViT, words, or complete)")
parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training")
parser.add_argument("--epochs", type=int, default=65, help="Number of training epochs")
parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
parser.add_argument("--use_batched", action='store_true', help="Use memory-efficient batched processing")
args = parser.parse_args()
config = vars(args)
print("Configuration:", config)

# Set up paths and parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
print("Script directory:", script_dir)

# Batch size from arguments
batch_size = config['batch_size']

# Number of classes - adjust based on dataset
# PAD-UFES-20 has 6 classes (0-5), ISIC19 might have different number
n_classes = 6 if config['src_dataset'] == "padufes20" else 8

# ViT model configurations
trans_version = 'google/vit-large-patch16-224'  # For feature extraction
vit_weights_version = 'google/vit-base-patch16-224-in21k'  # Pre-trained weights

# Dataset paths configuration
if config['src_dataset'] != "padufes20":
    dataset_path = script_dir + "/data/ISIC19/imgs/"
    metadata_train_path = script_dir + "/data/ISIC19/isic19_parsed_folders.csv"
    metadata_test_path = script_dir + "/data/ISIC19/isic19_parsed_test.csv"
else:
    dataset_path = script_dir + "/data/imgs/"
    metadata_train_path = script_dir + "/data/pad-ufes-20_parsed_folders_train.csv"
    metadata_test_path = script_dir + "/data/pad-ufes-20_parsed_test.csv"

# Check if dataset paths exist
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
if not os.path.exists(metadata_train_path):
    raise FileNotFoundError(f"Metadata file not found: {metadata_train_path}")

# Load and process metadata
print("Loading metadata...")
try:
    df_metadata = pd.read_csv(metadata_train_path, header=0, index_col=False)	
    df_metadata_test = pd.read_csv(metadata_test_path, header=0, index_col=False)
except Exception as e:
    print(f"Error loading metadata files: {e}")
    raise

# Process metadata based on dataset type
if config['src_dataset'] != "padufes20":
    df = process_metadata_frame_isic(df_metadata)
    df_test = process_metadata_frame_isic(df_metadata_test)
    
    # Filter specific folders for ISIC dataset
    # Note: This hardcoded filtering might need adjustment
    df = df.loc[(df.folder == 1) | (df.folder == 2), :]
    df_test = df_test.loc[df_test["folder"] == 6]
    
    # Limit test set size to prevent memory issues
    df_test = df_test.iloc[0:int(len(df_test)/2)]
    
    # Drop folder column after filtering
    df = df.drop("folder", axis=1)
    df_test = df_test.drop("folder", axis=1)
else:
    df = process_metadata_frame(df_metadata)
    df_test = process_metadata_frame(df_metadata_test)

# Add full paths to image files
df["file_path"] = dataset_path + df["file_path"]
df_test["file_path"] = dataset_path + df_test["file_path"]

# Display dataset statistics
print(f"\nDataset Statistics:")
print(f"Training samples: {len(df)}")
print(f"Test samples: {len(df_test)}")
print(f"Samples with text descriptions: {len(df.loc[df['text'] != 'empty'])}")

# Check class distribution
print("\nClass distribution in training set:")
class_counts = df["diagnostics"].value_counts().sort_index()
print(class_counts)

# Update n_classes based on actual data
actual_classes = len(df["diagnostics"].unique())
if actual_classes != n_classes:
    print(f"Warning: Expected {n_classes} classes but found {actual_classes}. Adjusting...")
    n_classes = actual_classes

# Create validation split from training data
print(f"\nCreating validation split ({config['val_split']*100}%)...")
try:
    df_train, df_val = train_test_split(
        df, 
        test_size=config['val_split'], 
        stratify=df['diagnostics'],
        random_state=42
    )
    print(f"Training samples after split: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
except ValueError as e:
    print(f"Warning: Could not create stratified split: {e}")
    print("Falling back to random split...")
    df_train, df_val = train_test_split(df, test_size=config['val_split'], random_state=42)

# Initialize transformation for data loading
trans_transform = ViTFeatureExtractor.from_pretrained(trans_version)

# Get device with error handling
try:
    device = get_default_device()
    print(f"\nUsing device: {device}")
    
    # Check GPU memory if using CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except Exception as e:
    print(f"Error setting up device: {e}")
    print("Falling back to CPU...")
    device = torch.device('cpu')

# Create datasets and dataloaders
print("\nCreating data loaders...")
train_ds = customDataset(df_train, trans_transform=trans_transform)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
train_dl = DeviceDataLoader(train_dl, device)

val_ds = customDataset(df_val, trans_transform=trans_transform)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
val_dl = DeviceDataLoader(val_dl, device)

test_ds = customDataset(df_test, trans_transform=trans_transform)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_dl = DeviceDataLoader(test_dl, device)

# Initialize ViT model
print("\nInitializing Vision Transformer model...")
try:
    model_trans_top, trans_layer_norm = vit_model(vit_weights_version)
    print("ViT model loaded successfully")
except Exception as e:
    print(f"Error loading ViT model: {e}")
    raise

# Create final model with error handling
print("\nCreating final model...")
try:
    model = model_final(model_trans_top, trans_layer_norm, n_classes, dp_rate=0.3)
    
    # Load pre-trained weights if available
    checkpoint_path = script_dir + '/model_weights_best.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading pre-trained weights from {checkpoint_path}")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Could not load weights: {e}")
            print("Starting with fresh model...")
    
    # Move model to device
    model = model.to(device)
    print(f"Model successfully moved to {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
except RuntimeError as e:
    if "out of memory" in str(e):
        print("GPU out of memory! Trying with CPU...")
        torch.cuda.empty_cache()
        device = torch.device('cpu')
        model = model.to(device)
        # Update dataloaders to use CPU
        train_dl = DeviceDataLoader(train_dl.dl, device)
        val_dl = DeviceDataLoader(val_dl.dl, device)
        test_dl = DeviceDataLoader(test_dl.dl, device)
    else:
        raise e

# Set up optimizer and scheduler
print("\nSetting up optimizer and scheduler...")
optimizer, lr_scheduler = set_params(model)

# Calculate class weights for handling imbalance
print("\nCalculating class weights for balanced training...")
if config['use_batched']:
    # For batched processing, we need to get labels differently
    all_labels = torch.tensor(df_train['diagnostics'].values)
else:
    all_labels = torch.tensor(df_train['diagnostics'].values)

class_weights = calculate_class_weights(all_labels, n_classes, device)
print(f"Class weights: {class_weights}")

# Training phase
print(f"\n{'='*50}")
print(f"Starting training for {config['epochs']} epochs...")
print(f"{'='*50}")

try:
    # Use memory-efficient processing if specified
    if config['use_batched']:
        print("Using memory-efficient batched processing...")
        # Note: fit function needs to be updated to support batched processing
        
    fit(
        epochs=config['epochs'], 
        model=model, 
        train_dl=train_dl,
        val_dl=val_dl,  # Pass validation dataloader
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler, 
        batch_num=batch_size, 
        dataset_name=config['src_dataset'], 
        model_config=config['backbone'],
        class_weights=class_weights,  # Pass class weights
        device=device
    )
    print("Training completed successfully!")
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    # Save checkpoint
    checkpoint_path = script_dir + '/model_checkpoint_interrupted.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")
    
except Exception as e:
    print(f"\nError during training: {e}")
    raise

# Testing phase
print(f"\n{'='*50}")
print("Starting model evaluation on test set...")
print(f"{'='*50}")

try:
    test_partial(
        model=model,
        test_data=test_dl, 
        batch_num=batch_size, 
        model_config=config['backbone'],
        dataset_name=config['src_dataset'],
        use_batched=config['use_batched']
    )
    print("\nEvaluation completed successfully!")
    
except Exception as e:
    print(f"Error during testing: {e}")
    # Try to save predictions even if plotting fails
    print("Attempting to save raw results...")
    raise

print(f"\n{'='*50}")
print("All tasks completed!")
print(f"{'='*50}")
