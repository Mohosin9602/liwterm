import os
import pandas as pd
import numpy as np
import torch
import tiktoken

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTModel, ViTConfig, AutoConfig, AutoTokenizer, AutoModel, BertModel, AutoModelForSequenceClassification, pipeline

# Allow loading of truncated images to prevent errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

################################################################################################################################################################################################################
####################################################################### Dataloader #############################################################################################################################
################################################################################################################################################################################################################
 

class customDataset(Dataset):
    """
    Custom dataset class for loading images with text descriptions and labels.
    
    Args:
        dataframe: pandas DataFrame containing file paths, text descriptions, and diagnostic labels
        trans_transform: transformation to apply to images (e.g., ViT feature extractor)
        text_transform: transformation to apply to text (currently unused)
    """
    def __init__(self, dataframe, trans_transform=None, text_transform=None):
        self.labels = dataframe["diagnostics"]
        self.images = dataframe["file_path"]
        self.text = dataframe["text"]
        self.trans_transform = trans_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Returns raw data for a single sample. Actual transformations are applied later
        to enable batch processing and memory efficiency.
        """
        # Get the image path, text, and label for this index
        img_path = self.images.iloc[idx] if hasattr(self.images, 'iloc') else self.images[idx]
        text = self.text.iloc[idx] if hasattr(self.text, 'iloc') else self.text[idx]
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        # Return raw data - transformations will be applied in process_data_batched
        return img_path, text, label

################################################################################################################################################################################################################
####################################################################### Data Processing ########################################################################################################################
################################################################################################################################################################################################################

# PAD-UFES-20 Processing 

def process_metadata_frame(df_meta):
	"""
	Process PAD-UFES-20 metadata into a format suitable for training.
	Converts binary features into natural language descriptions.
	
	Args:
	    df_meta: raw metadata DataFrame
	
	Returns:
	    df: processed DataFrame with file_path, text descriptions, and diagnostic labels
	"""
	#create the data frame
	df = pd.DataFrame()
	df["file_path"] = list(df_meta["img_id"])
	
	# Convert image IDs to proper file paths
	for i in range(len(df)):
		df.at[i,"file_path"] = str(df.iloc[i]["file_path"]).split(".")[0] + ".png"
	
	# Initialize columns
	df["text"] = "empty"
	df["diagnostics"] = "UNK"
	df["diagnostics_class"] = "UNK"	
	
	# Convert binary gender indicator to text
	df_meta.loc[df_meta["gender_FEMALE"] == 1, "gender_FEMALE"] = " The subject is a female."
	df_meta.loc[df_meta["gender_FEMALE"] == 0, "gender_FEMALE"] = " The subject is a male. "

	# Convert medical history indicators to text
	df_meta.loc[df_meta["skin_cancer_history_True"] == 1, "skin_cancer_history_True"] = " There is a skin cancer history."
	df_meta.loc[df_meta["skin_cancer_history_True"] == 0, "skin_cancer_history_True"] = " There is no skin cancer history."

	df_meta.loc[df_meta["cancer_history_True"] == 1, "cancer_history_True"] = " There is a cancer history."
	df_meta.loc[df_meta["cancer_history_True"] == 0, "cancer_history_True"] = " There is no cancer history."

	# Process Fitzpatrick skin type
	df_meta["fitspatrick_index"] = " No fitspatrick available."
	for fits in ["fitspatrick_1.0", "fitspatrick_2.0", "fitspatrick_3.0", "fitspatrick_4.0", "fitspatrick_5.0", "fitspatrick_6.0"]:
		for i in range(len(df_meta)):
			if df_meta.at[i,fits] == 1:
				df_meta.at[i,"fitspatrick_index"] = " " + fits
	
	# Process body region information			
	df_meta["cancer_region"] = " No region available."		
	for regs in ["region_ARM", "region_NECK", "region_FACE", "region_HAND", "region_FOREARM", "region_CHEST", "region_NOSE", "region_THIGH", "region_SCALP", "region_EAR", "region_BACK", "region_FOOT", "region_ABDOMEN", "region_LIP"]:
		for i in range(len(df_meta)):
			if df_meta.at[i,regs] == 1:
				df_meta.at[i,"cancer_region"] = " Lesion located in the region of the " + regs.split("_")[1] + "."

	# Process symptom indicators
	df_meta.loc[df_meta["itch_True"] == 1, "itch_True"] = " The lesion itches."
	df_meta.loc[df_meta["itch_True"] == 0, "itch_True"] = " The lesion does not itch."

	df_meta.loc[df_meta["grew_True"] == 1, "grew_True"] = " The lesion has grown."
	df_meta.loc[df_meta["grew_True"] == 0, "grew_True"] = " The lesion did not grow."

	df_meta.loc[df_meta["hurt_True"] == 1, "hurt_True"] = " The lesion hurts."
	df_meta.loc[df_meta["hurt_True"] == 0, "hurt_True"] = " The lesion does not hurt."

	df_meta.loc[df_meta["changed_True"] == 1, "changed_True"] = " The lesion has changed over time."
	df_meta.loc[df_meta["changed_True"] == 0, "changed_True"] = " The lesion did not change."

	df_meta.loc[df_meta["bleed_True"] == 1, "bleed_True"] = " The lesion bleeds."
	df_meta.loc[df_meta["bleed_True"] == 0, "bleed_True"] = " The lesion does not bleed."

	df_meta.loc[df_meta["elevation_True"] == 1, "elevation_True"] = " The lesion presents elevation."
	df_meta.loc[df_meta["elevation_True"] == 0, "elevation_True"] = " The lesion does not present elevation."
	
	# Combine all features into text descriptions
	for i in range(len(df)):
	  # Match patient and lesion IDs
	  for j in range(len(df_meta)):
	    if ((str(df.iloc[i]["file_path"].split('_')[1]) == str(df_meta.iloc[j]["patient_id"].split('_')[1])) and (str(df.iloc[i]["file_path"].split('_')[2]) == str(df_meta.iloc[j]["lesion_id"]))):
	      # Create comprehensive text description
	      df.at[i,"text"] = "Age of " + str(df_meta.iloc[j]["age"]) + "." + str(df_meta.iloc[j]["gender_FEMALE"]) + str(df_meta.iloc[j]["skin_cancer_history_True"]) + str(df_meta.iloc[j]["cancer_history_True"]) + str(df_meta.iloc[j]["fitspatrick_index"]) + str(df_meta.iloc[j]["cancer_region"]) + str(df_meta.iloc[j]["itch_True"]) + str(df_meta.iloc[j]["grew_True"]) + str(df_meta.iloc[j]["hurt_True"]) + str(df_meta.iloc[j]["changed_True"]) + str(df_meta.iloc[j]["bleed_True"]) + str(df_meta.iloc[j]["elevation_True"])
	      df.at[i,"diagnostics"] = str(df_meta.iloc[j]["diagnostic"])
	      df.at[i,"diagnostics_class"] = str(df_meta.iloc[j]["diagnostic"])
	      
	# Convert diagnostic labels to numeric values
	# Note: This mapping should be consistent across train and test sets
	df.loc[df["diagnostics"] == "NEV", "diagnostics"] = 0  # Nevus
	df.loc[df["diagnostics"] == "BCC", "diagnostics"] = 1  # Basal Cell Carcinoma
	df.loc[df["diagnostics"] == "ACK", "diagnostics"] = 2  # Actinic Keratosis
	df.loc[df["diagnostics"] == "SEK", "diagnostics"] = 3  # Seborrheic Keratosis
	df.loc[df["diagnostics"] == "SCC", "diagnostics"] = 4  # Squamous Cell Carcinoma
	df.loc[df["diagnostics"] == "BOD", "diagnostics"] = 4  # Bowenoid (mapped to SCC)
	df.loc[df["diagnostics"] == "MEL", "diagnostics"] = 5  # Melanoma
	
	return(df)  

# ISIC19 Processing

def process_metadata_frame_isic(df_meta):
	"""
	Process ISIC19 metadata into a format suitable for training.
	Similar to PAD-UFES-20 but with different feature structure.
	"""
	#create the data frame
	df = pd.DataFrame()
	df["file_path"] = list(df_meta["img_id"])
	df["folder"] = list(df_meta["folder"])
	
	# Convert to JPEG extension
	for i in range(len(df)):
		df.at[i,"file_path"] = str(df.iloc[i]["file_path"]) + ".jpg"
	
	# Initialize columns
	df["text"] = "empty"
	df["diagnostics_class"] = df_meta["diagnostic"]
	df["age"] = df_meta["age"]
	df["diagnostics"] = df_meta["diagnostic_number"]
     
	# Copy region and gender columns
	for regs in ["region_anterior torso", "region_upper extremity", "region_posterior torso", "region_lower extremity", "region_lateral torso", "region_head/neck", "region_palms/soles", "region_oral/genital"]:
		df[regs] = df_meta[regs]
          
	for genders in ["gender_male", "gender_female"]:
		df[genders] = df_meta[genders]  # Fixed: Changed from df_meta[regs] to df_meta[genders]

	# Process region information
	df["region"] = " No region available."		
	for regs in ["region_anterior torso", "region_upper extremity", "region_posterior torso", "region_lower extremity", "region_lateral torso", "region_head/neck", "region_palms/soles", "region_oral/genital"]:
		for i in range(len(df)):
			if df.at[i,regs] == 1:
				df.at[i,"region"] = " Lesion located in the region of the " + regs.split("_")[1] + "."
 
	# Create text descriptions
	for i in range(len(df)):
            # Process gender
            if df.iloc[i, df.columns.get_loc("gender_female")] == 0 and df.iloc[i, df.columns.get_loc("gender_male")] == 1:
                 gender =  " The subject is a male."
            elif df.iloc[i, df.columns.get_loc("gender_female")] == 1 and df.iloc[i, df.columns.get_loc("gender_male")] == 0:
                 gender =  " The subject is a female."
            else:
                 gender =  " No gender available."
            
            # Process age
            if (df.iloc[i, df.columns.get_loc("age")]) == 0:
                 age = "Age not available."
            else:
                 age = "Age of " + str(int(df.iloc[i]["age"])) + "."
            
            # Combine features
            df.at[i,"text"] = age + str(df.iloc[i, df.columns.get_loc("region")]) + gender
	      
	# Drop temporary columns
	df = df.drop(columns=["region_anterior torso", "region_upper extremity", "region_posterior torso", "region_lower extremity", "region_lateral torso", "region_head/neck", "region_palms/soles", "region_oral/genital", "age", "gender_female", "gender_male"])
	
	return(df)  


################################################################################################################################################################################################################
####################################################################### Feature Calculation ####################################################################################################################
################################################################################################################################################################################################################
 
# Initialize feature extractors and tokenizers
feature_extractor_text = pipeline("feature-extraction",framework="pt",model="facebook/bart-base")
tokenizer = AutoTokenizer.from_pretrained("BEE-spoke-data/cl100k_base-mlm")
trans_transform = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')
tiktokenizer = tiktoken.get_encoding("cl100k_base")

def process_data_batched(dataloader, dataset_name, device, batch_size=32):
    """
    Memory-efficient batch processing of data.
    Processes images and text in smaller batches to avoid GPU memory issues.
    
    Args:
        dataloader: DataLoader containing the dataset
        dataset_name: name of the dataset ('padufes20' or other)
        device: torch device (cuda or cpu)
        batch_size: size of batches for processing (default: 32)
    
    Returns:
        tuple: (image_features, text_features, labels) all as torch tensors
    """
    all_image_features = []
    all_text_features = []
    all_labels = []
    
    # Process data in batches to avoid memory issues
    dataset = dataloader.dataset
    total_samples = len(dataset)
    
    print(f"Processing {total_samples} samples in batches of {batch_size}...")
    
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_size_actual = end_idx - start_idx
        
        # Get batch data
        batch_img_paths = []
        batch_texts = []
        batch_labels = []
        
        for idx in range(start_idx, end_idx):
            img_path, text, label = dataset[idx]
            batch_img_paths.append(img_path)
            batch_texts.append(text)
            batch_labels.append(label)
        
        try:
            # Process images
            images = []
            for img_path in batch_img_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(np.array(img))
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    # Create blank image as fallback
                    images.append(np.zeros((224, 224, 3), dtype=np.uint8))
            
            # Apply transformations
            image_features = trans_transform(images, return_tensors='pt')['pixel_values']
            
            # Process text
            text_features_list = []
            text_outputs = feature_extractor_text(batch_texts, return_tensors="pt")
            for i in range(len(text_outputs)):
                text_features_list.append(torch.from_numpy(text_outputs[i][0].numpy().mean(axis=0)))
            text_features = torch.stack(text_features_list)
            
            # Move to device and append
            all_image_features.append(image_features.to(device))
            all_text_features.append(text_features.to(device))
            all_labels.extend(batch_labels)
            
            print(f"Processed batch {start_idx//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU OOM at batch starting at {start_idx}. Trying with smaller batch...")
                # Clear cache and try with smaller batch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                # Recursively call with half batch size
                if batch_size > 1:
                    return process_data_batched(dataloader, dataset_name, device, batch_size//2)
                else:
                    raise e
            else:
                raise e
    
    # Concatenate all batches
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.float64).to(device)
    
    return all_image_features, all_text_features, all_labels

def process_data(Dataset, dataset_name):
    """
    Legacy function for processing data. Loads all images at once.
    WARNING: This can cause memory issues with large datasets.
    Consider using process_data_batched instead.
    """
    #getting the data
    start = 0
    end = 0 + int(len(Dataset.dataset.images)/6)
    iteration = int(len(Dataset.dataset.images)/6)

    input_trans = torch.tensor([])
    l = []
    yb = torch.tensor([])
    
    if dataset_name != "padufes20":
        while(end <= len(Dataset.dataset.images)): 
            input_trans = torch.cat((input_trans, (trans_transform([np.array(Image.open(x).convert('RGB')) for x in Dataset.dataset.images[start:end]], return_tensors='pt'))['pixel_values'].squeeze()))
            input_text = list(feature_extractor_text(list(Dataset.dataset.text[start:end]), return_tensors="pt"))
            for i in range(len(input_text)):
                l.append(torch.from_numpy(input_text[i][0].numpy().mean(axis=0)))
            yb = torch.cat((yb, torch.tensor((Dataset.dataset.labels[start:end]).to_numpy(dtype=np.float64))))
            start = end
            end += iteration
            if start < len(Dataset.dataset.images)-10 and end > len(Dataset.dataset.images):
                end = len(Dataset.dataset.images) 
            print(start)
            print(end)
  else:              
    
    #for the PAD-UFES-20 dataset
    input_trans = (trans_transform([np.array(Image.open(x).convert('RGB')) for x in Dataset.dataset.images], return_tensors='pt'))['pixel_values'].squeeze()
    l = []
    input_text = list(feature_extractor_text(list(Dataset.dataset.text), return_tensors="pt"))
    for i in range(len(input_text)):
        l.append(torch.from_numpy(input_text[i][0].numpy().mean(axis=0)))
    yb = torch.tensor((Dataset.dataset.labels[:]).to_numpy(dtype=np.float64))
    
  return(input_trans, torch.stack(l), yb)	

def process_data_2(Dataset):
    """
    Alternative data processing using tiktoken tokenizer.
    Currently unused but kept for reference.
    """
    #getting the data
    input_trans = (trans_transform([np.array(Image.open(x).convert('RGB')) for x in Dataset.dataset.images], return_tensors='pt'))['pixel_values'].squeeze()
    l = []
    l_input_text = list(Dataset.dataset.text)
    for i in l_input_text:
       tokens = list(tiktokenizer.encode(i))
       # Pad tokens to fixed length
       while len(tokens) < 90:
            tokens.append(0)
       l.append(torch.tensor(tokens).bfloat16())		
    	     
    yb = torch.tensor((Dataset.dataset.labels[:]).to_numpy(dtype=np.float64))
    return(input_trans, torch.stack(l), yb)

################################################################################################################################################################################################################
####################################################################### Add Data/Model to GPU ##################################################################################################################
################################################################################################################################################################################################################

def get_default_device():
    """
    Get the default device for computation.
    Returns CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """
    Move data to specified device.
    Handles lists and tuples recursively.
    """
    # if data is list or tuple, move each of them to device
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """
    Wrapper for DataLoader that automatically moves batches to specified device.
    This ensures all data is on the correct device (GPU/CPU) during iteration.
    """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        self.dataset = dl.dataset  # Expose dataset for compatibility
        
    def __iter__(self):
        """Yield batches moved to the specified device"""
        for b in self.dl:
            yield to_device(b, self.device)
            
    def __len__(self):
        return len(self.dl)

################################################################################################################################################################################################################
####################################################################### Metrics/Params #########################################################################################################################
################################################################################################################################################################################################################
        
def accuracy(predictions, labels):
    """
    Calculate accuracy for multi-class classification.
    
    Args:
        predictions: model output logits (batch_size, n_classes)
        labels: true labels (batch_size,)
    
    Returns:
        float: accuracy as a fraction between 0 and 1
    """
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())
    
# Define optimizer and learning_rate scheduler

def set_params(model):
    """
    Set up optimizer and learning rate scheduler for training.
    
    Args:
        model: the neural network model
    
    Returns:
        tuple: (optimizer, lr_scheduler)
    """
    # Get only parameters that require gradients
    params = [param for param in list(model.parameters()) if param.requires_grad]
    
    # SGD optimizer with momentum
    optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.2)

    # Learning rate scheduler that reduces LR when loss plateaus
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # Minimize loss
        factor=0.1,        # Reduce LR by 10x
        patience=4,        # Wait 4 epochs before reducing
        min_lr=1e-6       # Don't go below this LR
    )
    return optimizer, lr_scheduler

# Global batch size variable (unused, consider removing)
batch_num = 24    

# Helper function to calculate class weights for imbalanced datasets
def calculate_class_weights(labels, n_classes, device):
    """
    Calculate class weights for weighted loss function.
    Helps with class imbalance by giving more weight to rare classes.
    
    Args:
        labels: tensor of labels
        n_classes: number of classes
        device: torch device
    
    Returns:
        tensor: weights for each class
    """
    class_counts = torch.zeros(n_classes)
    for i in range(n_classes):
        class_counts[i] = (labels == i).sum()
    
    # Avoid division by zero
    class_counts = torch.maximum(class_counts, torch.tensor(1.0))
    
    # Calculate inverse frequency weights
    total_samples = len(labels)
    class_weights = total_samples / (n_classes * class_counts)
    
    return class_weights.to(device)

