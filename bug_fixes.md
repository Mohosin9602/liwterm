# Bug Fixes and Improvements for LiwTerm Project

## Critical Fixes Applied:

### 1. ✅ Confusion Matrix Error
**Fixed:** The plot_confusion_matrix function now handles variable number of classes correctly and saves to file instead of displaying (better for remote environments).

### 2. ✅ Batch Size Inconsistency  
**Fixed:** DataLoaders now use the defined batch_size variable (24) instead of hardcoded 16.

### 3. ✅ Duplicate Code in set_params
**Fixed:** Removed unreachable duplicate code in the set_params function.

## Remaining Issues to Address:

### 1. 🔴 Path Consistency
```python
# Fix in main.py:
if config['src_dataset'] != "padufes20":
    dataset_path = script_dir + "/data/ISIC19/imgs/"  # Add script_dir
else:
    dataset_path = script_dir + "/data/imgs/"
```

### 2. 🔴 Model Weight Loading
```python
# Add map_location for cross-device compatibility:
model.load_state_dict(torch.load('model_weights_1228', map_location=device))
```

### 3. 🔴 Memory-Efficient Data Processing
Instead of loading all images at once, process in batches:
```python
def process_data_batched(dataloader, dataset_name, device):
    all_images = []
    all_texts = []
    all_labels = []
    
    for batch in dataloader:
        img_paths, texts, labels = batch
        # Process batch
        images = trans_transform([Image.open(p).convert('RGB') for p in img_paths])
        all_images.append(images['pixel_values'].to(device))
        # ... process texts and labels
    
    return torch.cat(all_images), torch.cat(all_texts), torch.cat(all_labels)
```

### 4. 🔴 Class Imbalance
The model never predicts class 5, indicating training issues. Consider:
- Checking class distribution in training data
- Using weighted loss function
- Oversampling minority classes

### 5. 🟡 Add Error Handling
```python
try:
    model = model_final(model_trans_top, trans_layer_norm, n_classes, dp_rate=0.3)
    model = model.to(device)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("GPU out of memory, falling back to CPU")
        device = torch.device('cpu')
        model = model.to(device)
    else:
        raise e
```

### 6. 🟡 Requirements with Versions
Create requirements.txt:
```
transformers==4.30.0
torch==2.0.1
torchvision==0.15.2
datasets==2.12.0
torchaudio==2.0.2
sentencepiece==0.1.99
GPUtil==1.4.0
tiktoken==0.4.0
pandas==2.0.2
numpy==1.24.3
matplotlib==3.7.1
Pillow==9.5.0
```

### 7. 🟡 Validation Set Implementation
Split training data for validation:
```python
from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df, test_size=0.2, stratify=df['diagnostics'])
```

### 8. 🟡 Early Stopping
Add early stopping to prevent overfitting:
```python
best_val_loss = float('inf')
patience_counter = 0
for epoch in range(epochs):
    # ... training ...
    val_loss = validate(model, val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter > patience:
            print("Early stopping triggered")
            break
```

## Recommended Workflow:

1. Fix path issues first
2. Add proper error handling
3. Implement validation split
4. Add memory-efficient data loading
5. Address class imbalance
6. Add comprehensive logging

## Quick Test Command:
```bash
# Test with smaller batch size to avoid memory issues
python main.py padufes20 complete
``` 