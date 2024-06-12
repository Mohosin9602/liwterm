import os
import pandas as pd
import numpy as np
import torch
import tiktoken

from PIL import Image, ImageFile
from datasets import Dataset
from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTModel, ViTConfig, AutoConfig, AutoTokenizer, AutoModel, BertModel, AutoModelForSequenceClassification, pipeline

ImageFile.LOAD_TRUNCATED_IMAGES = True

################################################################################################################################################################################################################
####################################################################### Dataloader #############################################################################################################################
################################################################################################################################################################################################################
 

class customDataset(Dataset):
    def __init__(self, dataframe, trans_transform=None, text_transform=None):
        self.labels = dataframe["diagnostics"]
        self.images = dataframe["file_path"]
        self.text = dataframe["text"]
        self.trans_transform = trans_transform

    def __len__ (self):
        return len(self.labels)
    '''
    def __getitem__(self, idx):

        img_path = self.images[idx]
        tmp = []
        print(idx)
        for i in (idx):
          #print(img_path[i])
          image = Image.open(img_path[i]).convert('RGB')
          image_trans = self.trans_transform(np.array(image), return_tensors='pt')
          image_trans = image_trans['pixel_values'].squeeze()
          tmp.append(image_trans)

        img_trans_out = pd.Series(tmp, index = idx)
        #print(tmp)

        print(type(img_trans_out))
        text = self.text[idx]
        print(type(text))

        label = self.labels[idx]
        print(type(label))

        return img_trans_out, text, label
     '''

################################################################################################################################################################################################################
####################################################################### Data Processing ########################################################################################################################
################################################################################################################################################################################################################

# PAD-UFES-20 Processing 

def process_metadata_frame(df_meta):
	#create the data frame
	df = pd.DataFrame()
	df["file_path"] = list(df_meta["img_id"])
	#df["folder"] = list(df_meta["folder"])
	for i in range(len(df)):
		df.at[i,"file_path"] = str(df.iloc[i]["file_path"]).split(".")[0] + ".png"
	df["text"] = "empty"
	df["diagnostics"] = "UNK"
	df["diagnostics_class"] = "UNK"	
	
	#metadata processing
	df_meta.loc[df_meta["gender_FEMALE"] == 1, "gender_FEMALE"] = " The subject is a female."
	df_meta.loc[df_meta["gender_FEMALE"] == 0, "gender_FEMALE"] = " The subject is a male. "

	df_meta.loc[df_meta["skin_cancer_history_True"] == 1, "skin_cancer_history_True"] = " There is a skin cancer history."
	df_meta.loc[df_meta["skin_cancer_history_True"] == 0, "skin_cancer_history_True"] = " There is no skin cancer history."

	df_meta.loc[df_meta["cancer_history_True"] == 1, "cancer_history_True"] = " There is a cancer history."
	df_meta.loc[df_meta["cancer_history_True"] == 0, "cancer_history_True"] = " There is no cancer history."


	df_meta["fitspatrick_index"] = " No fitspatrick available."
	for fits in ["fitspatrick_1.0", "fitspatrick_2.0", "fitspatrick_3.0", "fitspatrick_4.0", "fitspatrick_5.0", "fitspatrick_6.0"]:
		for i in range(len(df_meta)):
			if df_meta.at[i,fits] == 1:
				df_meta.at[i,"fitspatrick_index"] = " " + fits
				
	df_meta["cancer_region"] = " No region available."		
	for regs in ["region_ARM", "region_NECK", "region_FACE", "region_HAND", "region_FOREARM", "region_CHEST", "region_NOSE", "region_THIGH", "region_SCALP", "region_EAR", "region_BACK", "region_FOOT", "region_ABDOMEN", "region_LIP"]:
		for i in range(len(df_meta)):
			if df_meta.at[i,regs] == 1:
				df_meta.at[i,"cancer_region"] = " Lesion located in the region of the " + regs.split("_")[1] + "."

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
	
	#process the data frame
	for i in range(len(df)):
	  #print("patient_id: ",df.iloc[i]["file_path"].split('_')[1], "lesion_id: ", df.iloc[i]["file_path"].split('_')[2])
	  for j in range(len(df_meta)):
	    #print("meta_patient_id: " ,df_meta.iloc[j]["patient_id"].split('_')[1], "lesion_id: ", df.iloc[i]["file_path"].split('_')[2])
	    if ((str(df.iloc[i]["file_path"].split('_')[1]) == str(df_meta.iloc[j]["patient_id"].split('_')[1])) and (str(df.iloc[i]["file_path"].split('_')[2]) == str(df_meta.iloc[j]["lesion_id"]))):
	      #print("meta_patient_id: " ,df_meta.iloc[j]["patient_id"].split('_')[1], "meta_lesion_id: ", df_meta.iloc[i]["lesion_id"])
	      df.at[i,"text"] = "Age of " + str(df_meta.iloc[j]["age"]) + "." + str(df_meta.iloc[j]["gender_FEMALE"]) + str(df_meta.iloc[j]["skin_cancer_history_True"]) + str(df_meta.iloc[j]["cancer_history_True"]) + str(df_meta.iloc[j]["fitspatrick_index"]) + str(df_meta.iloc[j]["cancer_region"]) + str(df_meta.iloc[j]["itch_True"]) + str(df_meta.iloc[j]["grew_True"]) + str(df_meta.iloc[j]["hurt_True"]) + str(df_meta.iloc[j]["changed_True"]) + str(df_meta.iloc[j]["bleed_True"]) + str(df_meta.iloc[j]["elevation_True"])
	      df.at[i,"diagnostics"] = str(df_meta.iloc[j]["diagnostic"])
	      df.at[i,"diagnostics_class"] = str(df_meta.iloc[j]["diagnostic"])
	      
	#set the diagnostics label
	df.loc[df["diagnostics"] == "NEV", "diagnostics"] = 0
	df.loc[df["diagnostics"] == "BCC", "diagnostics"] = 1
	df.loc[df["diagnostics"] == "ACK", "diagnostics"] = 2
	df.loc[df["diagnostics"] == "SEK", "diagnostics"] = 3
	df.loc[df["diagnostics"] == "SCC", "diagnostics"] = 4
	df.loc[df["diagnostics"] == "BOD", "diagnostics"] = 4
	df.loc[df["diagnostics"] == "MEL", "diagnostics"] = 5   
	
	return(df, df_meta)  

# ISIC19 Processing

def process_metadata_frame_isic(df_meta):
	#create the data frame
	df = pd.DataFrame()
	df["file_path"] = list(df_meta["img_id"])
	df["folder"] = list(df_meta["folder"])
	for i in range(len(df)):
		df.at[i,"file_path"] = str(df.iloc[i]["file_path"]) + ".jpg"
	df["text"] = "empty"
	df["diagnostics_class"] = df_meta["diagnostic"]
	df["age"] = df_meta["age"]
	df["diagnostics"] = "UNK"
     
	for regs in ["region_anterior torso", "region_upper extremity", "region_posterior torso", "region_lower extremity", "region_lateral torso", "region_head/neck", "region_palms/soles", "region_oral/genital"]:
		df[regs] = df_meta[regs]
          
	for genders in ["gender_male", "gender_female"]:
		df[genders] = df_meta[regs]

	df["region"] = " No region available."		
	for regs in ["region_anterior torso", "region_upper extremity", "region_posterior torso", "region_lower extremity", "region_lateral torso", "region_head/neck", "region_palms/soles", "region_oral/genital"]:
		for i in range(len(df)):
			if df.at[i,regs] == 1:
				df.at[i,"region"] = " Lesion located in the region of the " + regs.split("_")[1] + "."
 

	for i in range(len(df)):
            if df.iloc[i, df.columns.get_loc("gender_female")] == 0 and df.iloc[i, df.columns.get_loc("gender_male")] == 1:
                 gender =  " The subject is a male."
            elif df.iloc[i, df.columns.get_loc("gender_female")] == 1 and df.iloc[i, df.columns.get_loc("gender_male")] == 0:
                 gender =  " The subject is a female."
            else:
                 gender =  " No gender available."
            if (df.iloc[i, df.columns.get_loc("age")]) == 0:
                 age = "Age not available."
            else:
                 age = "Age of " + str(int(df.iloc[i]["age"])) + "."
            df.at[i,"text"] = age + str(df.iloc[i, df.columns.get_loc("region")]) + gender
	      
	#set the diagnostics label
	df.loc[df["diagnostics"] == "MEL", "diagnostics"] = 0
	df.loc[df["diagnostics"] == "NV", "diagnostics"] = 1
	df.loc[df["diagnostics"] == "BCC", "diagnostics"] = 2
	df.loc[df["diagnostics"] == "AK", "diagnostics"] = 3
	df.loc[df["diagnostics"] == "BKL", "diagnostics"] = 4
	df.loc[df["diagnostics"] == "DF", "diagnostics"] = 5
	df.loc[df["diagnostics"] == "VASC", "diagnostics"] = 6
	df.loc[df["diagnostics"] == "SCC", "diagnostics"] = 7
	df.loc[df["diagnostics"] == "UNK", "diagnostics"] = 8

	df = df.drop(columns=["region_anterior torso", "region_upper extremity", "region_posterior torso", "region_lower extremity", "region_lateral torso", "region_head/neck", "region_palms/soles", "region_oral/genital", "age", "gender_female", "gender_male"])
	
	return(df)  


################################################################################################################################################################################################################
####################################################################### Feature Calculation ####################################################################################################################
################################################################################################################################################################################################################
 

feature_extractor_text = pipeline("feature-extraction",framework="pt",model="facebook/bart-base")
tokenizer = AutoTokenizer.from_pretrained("BEE-spoke-data/cl100k_base-mlm")
trans_transform = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')
tiktokenizer = tiktoken.get_encoding("cl100k_base")

def process_data(Dataset):
  #getting the data
  start = 0
  end = 0 + int(len(Dataset.dataset.images)/3)
  iteration = int(len(Dataset.dataset.images)/3)

  #print(Dataset.dataset.images[start:end])
  #print(Dataset.dataset.text[start:end])

  l_input_trans = []
  l = []
  yb = []

  while(end <= len(Dataset.dataset.images)): 
       input_trans = list(trans_transform([np.array(Image.open(x).convert('RGB')) for x in Dataset.dataset.images[start:end]], return_tensors='pt'))['pixel_values'].squeeze()
       input_text = list(feature_extractor_text(list(Dataset.dataset.text[start:end]), return_tensors="pt"))
       for i in range(len(input_text)):
            l.append(torch.from_numpy(input_text[i][0].numpy().mean(axis=0)))
            l_input_trans.append(input_trans)
       yb.append((Dataset.dataset.labels[start:end]).to_numpy(dtype=np.float64))
       start = end
       end += iteration
       if start < len(Dataset.dataset.images)-10 and end > len(Dataset.dataset.images):
            end = len(Dataset.dataset.images) 
       print(start)
       print(end)
       #x = input()
            
  '''
  #for the PAD-UFES-20 dataset
  input_trans = (trans_transform([np.array(Image.open(x).convert('RGB')) for x in Dataset.dataset.images], return_tensors='pt'))['pixel_values'].squeeze()
  #input_text = tokenizer(list(Dataset.dataset.text), padding=True, truncation=True, max_length = 25, return_tensors="pt")
  l = []
  input_text = list(feature_extractor_text(list(Dataset.dataset.text), return_tensors="pt"))
  for i in range(len(input_text)):
  	l.append(torch.from_numpy(input_text[i][0].numpy().mean(axis=0)))
  yb = torch.tensor((Dataset.dataset.labels[:]).to_numpy(dtype=np.float64))
  '''
  return(torch.stack(l_input_trans), torch.stack(l), torch.tensor(yb))	

def process_data_2(Dataset):
  #getting the data
  input_trans = (trans_transform([np.array(Image.open(x).convert('RGB')) for x in Dataset.dataset.images], return_tensors='pt'))['pixel_values'].squeeze()
  #input_text = tokenizer(list(Dataset.dataset.text), padding=True, truncation=True, max_length = 25, return_tensors="pt")
  l = []
  l_input_text = list(Dataset.dataset.text)
  for i in l_input_text:
       tokens = list(tiktokenizer.encode(i))
       while len(tokens) < 90:
            tokens.append(0)
       l.append(torch.tensor(tokens).bfloat16())		
    	     
  yb = torch.tensor((Dataset.dataset.labels[:]).to_numpy(dtype=np.float64))
  return(input_trans, torch.stack(l), yb)

################################################################################################################################################################################################################
####################################################################### Add Data/Model to GPU ##################################################################################################################
################################################################################################################################################################################################################

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
def to_device(data, device):
    # if data is list or tuple, move each of them to device
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device) -> None:
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            # yield only execuate when the function is called
            yield to_device(b, self. device)

    def __len__(self):
        return len(self.dl)

################################################################################################################################################################################################################
####################################################################### Metrics/Params #########################################################################################################################
################################################################################################################################################################################################################
        
def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())
    
# Define optimizer and learning_rate scheduler

def set_params(model):
	params = [param for param in list(model.parameters()) if param.requires_grad]
	optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.2)
	#optimizer = torch.optim.Adam(params, lr=1e-3)
	lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
	    optimizer,
	    mode='min',
	    factor=0.1,
	    patience=4,
	    min_lr = 1e-6,
	    verbose=True)
	return(optimizer, lr_scheduler)

batch_num = 24    

