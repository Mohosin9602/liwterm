# -*- coding: utf-8 -*-
"""TransFuse.ipynb
Luis Souza
la.souza@inf.ufes.br
"""

################################################################################################################################################################################################################
####################################################################### Requirements.txt #######################################################################################################################
################################################################################################################################################################################################################
#pip install git+https://github.com/huggingface/transformers datasets torchaudio sentencepiece GPUtil

################################################################################################################################################################################################################
####################################################################### Imports ################################################################################################################################
################################################################################################################################################################################################################
#imports

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time

from datasets import Dataset
from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTModel, ViTConfig, AutoConfig, AutoTokenizer, AutoModel, BertModel, AutoModelForSequenceClassification, pipeline
from PIL import Image, ImageFile
from sklearn.preprocessing import OneHotEncoder
import random
from random import seed
import GPUtil
# %matplotlib inline


ImageFile.LOAD_TRUNCATED_IMAGES = True
print(os.getcwd())

################################################################################################################################################################################################################
####################################################################### Data Loading and Processing ############################################################################################################
################################################################################################################################################################################################################

def process_metadata_frame(df_meta):
	#create the data frame
	df = pd.DataFrame()
	df["file_path"] = list(df_meta["img_id"])
	for i in range(len(df)):
		df.at[i,"file_path"] = str(df.iloc[i]["file_path"]).split(".")[0] + ".jpg"
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

################################################################################################################################################################################################################
####################################################################### Feature Extraction #####################################################################################################################
################################################################################################################################################################################################################	

feature_extractor_text = pipeline("feature-extraction",framework="pt",model="facebook/bart-base")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
trans_transform = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')

def process_data(Dataset):
  #getting the data
  input_trans = (trans_transform([np.array(Image.open(x).convert('RGB')) for x in Dataset.dataset.images], return_tensors='pt'))['pixel_values'].squeeze()
  #input_text = tokenizer(list(Dataset.dataset.text), padding=True, truncation=True, max_length = 25, return_tensors="pt")
  l = []
  input_text = list(feature_extractor_text(list(Dataset.dataset.text), return_tensors="pt"))
  for i in range(len(input_text)):
  	l.append(torch.from_numpy(input_text[i][0].numpy().mean(axis=0)))
  yb = torch.tensor(Dataset.dataset.labels[:])
  return(input_trans, torch.stack(l), yb)	
		
#Load training data
files = os.listdir("data/imgs_1_2_3/")
df_metadata = pd.read_csv("data/pad-ufes-20_parsed_folders_train.csv", header = 0, index_col = False)	
df_metadata_test = pd.read_csv("data/pad-ufes-20_parsed_test.csv", header = 0, index_col = False)

df,df_metadata = process_metadata_frame(df_metadata)
df["file_path"] = "data/imgs_1_2_3/" + df["file_path"]

print(len(df.loc[df["text"] != "empty"]))
print(df.loc[df["text"] != "empty"])

#Load Validation data
files_test = os.listdir("data/imgs_1_2_3/")

df_test,df_metadata_test = process_metadata_frame(df_metadata_test)
df_test["file_path"] = "data/imgs_1_2_3/" + df_test["file_path"]

print(len(df_test.loc[df_test["text"] != "empty"]))
print(df_test.loc[df_test["text"] != "empty"])


classes = tuple(df["diagnostics_class"].unique())
print(classes)
n_classes = 6


l = [df["text"][0], df["text"][1]]

# Define training dataset

################################################################################################################################################################################################################
####################################################################### Custom Data Loader Definition ##########################################################################################################
################################################################################################################################################################################################################

class customDataset(Dataset):
    def __init__(self, dataframe, trans_transform=None, text_transform=None):
        self.labels = dataframe["diagnostics"]
        self.images = dataframe["file_path"]
        self.text = dataframe["text"]
        self.trans_transform = trans_transform

    def __len__ (self):
        return len(self.labels)

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

#Loaders definition

train_ds = customDataset(df, trans_transform=trans_transform)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

test_ds = customDataset(df_test, trans_transform=trans_transform)
test_dl = DataLoader(test_ds, batch_size=16, shuffle=True)

print(test_dl.dataset.labels)

print

#print(train_dl.dataset.image_trans)

################################################################################################################################################################################################################
####################################################################### Models' Definitions ####################################################################################################################
################################################################################################################################################################################################################


# Modifying the ViT model - images

#print(model_trans)
model_trans = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
count = 0
for child in model_trans.children():
    #print(child, count)
    count += 1
    if count < 4:
        for param in child.parameters():
            param.requires_grad = False

layers_trans = list(model_trans.children()) # Get all the layers from the Transformer model
model_trans_top = nn.Sequential(*layers_trans[:-2]) # Remove the normalization layer and pooler layer
trans_layer_norm = list(model_trans.children())[2] # Get the normalization layer

#output = model_trans_top(inputs3)
#print(trans_layer_norm)

#Merging the two models

# Merge the two models
class model_final(nn.Module):
    def __init__(self, model_trans_top, trans_layer_norm, dp_rate = 0.3):
        super().__init__()
        # All the trans model layers
        self.model_trans_top = model_trans_top
        self.trans_layer_norm = trans_layer_norm
        self.trans_flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.trans_linear = nn.Linear(150528, 2048)
        self.batchnorm1 = nn.BatchNorm1d(4096)
        self.batchnorm2 = nn.BatchNorm1d(2048)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.batchnorm4 = nn.BatchNorm1d(512)
        
        # All the text model
        self.token_flatten = nn.Flatten()
        self.token_linear = nn.Linear(768, 2048)

        # Merge the result and pass the
        self.dropout = nn.Dropout(dp_rate)
        self.linear1 = nn.Linear(4096, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
        #to add classes, change 1 to n_classes
        #self.linear2 = nn.Linear(500,1)
        self.linear4 = nn.Linear(512,n_classes)

    #change forward to match with Bert Model
    def forward(self, trans_b, token_b_input_feats):
        # Get intermediate outputs using hidden layer
        result_trans = self.model_trans_top(trans_b)
        patch_state = result_trans.last_hidden_state[:,1:,:] # Remove the classification token and get the last hidden state of all patchs
        result_trans = self.trans_layer_norm(patch_state)
        result_trans = self.trans_flatten(patch_state)
        result_trans = self.dropout(result_trans)
        result_trans = self.trans_linear(result_trans)

        # Get intermediate outputs using hidden layer
        
        result_token = self.token_flatten(token_b_input_feats)
        result_token = self.dropout(result_token)
        result_token = self.token_linear(result_token)

        result_merge = torch.cat((result_trans, result_token),1)
        result_merge = self.batchnorm1(result_merge)
        result_merge = self.relu(result_merge)
        result_merge = self.dropout(result_merge)

        result_merge = self.linear1(result_merge)
        result_merge = self.batchnorm2(result_merge)
        result_merge = self.relu(result_merge)
        result_merge = self.dropout(result_merge)
        
        result_merge = self.linear2(result_merge)
        result_merge = self.batchnorm3(result_merge)
        result_merge = self.relu(result_merge)
        result_merge = self.dropout(result_merge)
        
        result_merge = self.linear3(result_merge)
        result_merge = self.batchnorm4(result_merge)
        result_merge = self.relu(result_merge)
        result_merge = self.dropout(result_merge)
        
        result_merge = self.linear4(result_merge)        

        return result_merge

model = model_final(model_trans_top, trans_layer_norm, dp_rate = 0.3)
# model.load_state_dict(torch.load('model_weights_1228'))

print(model)

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
####################################################################### Model's Parameters #####################################################################################################################
################################################################################################################################################################################################################
 
# Define optimizer and learning_rate scheduler
params = [param for param in list(model.parameters()) if param.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.2)
#optimizer = torch.optim.Adam(params, lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3,
    min_lr = 1e-6,
    verbose=True)

# fit and test #1

batch_num = 24

def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())

################################################################################################################################################################################################################
####################################################################### Model's Fit Function ###################################################################################################################
################################################################################################################################################################################################################

# Fit function doing grad steps
def fit(epochs, model, train_dl):
    opt = optimizer
    sched = lr_scheduler
    #loss_func = nn.MSELoss()
    loss_func = nn.CrossEntropyLoss()
    best_score = None
    patience = 50
    path = "sample_data/checkpoints/" # user_defined path to save model
    
    print("Calculating the features...")
    image_input,text_input,label = process_data(train_dl)
    
    print("Feature sizes: ViT({}); pipeline({}); labels({}).".format(image_input.size(), text_input.size(), label.size()))
    
    if not os.path.exists(path):
      # if the demo_folder directory is not present
      # then create it.
      os.makedirs(path)
    print("Training...\n")
    for epoch in range(epochs):
        #print("entrou treino")
        counter = 0
        model.train()
        #print("treinou")
        acc = 0
        total_loss = 0
        #label = label.float()

        #ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(label.reshape(-1,1))
        #print(ohe.categories_)
        #label = ohe.transform(label)
        #print(y)

        #print(type(image_input))
        #print(type(text_input['input_ids']))

        label = label.long()
        n_batches = int(int(label.size(dim=0))/batch_num)
        #number of batches
        for batches in range(int(int(label.size(dim=0))/batch_num)):
        #for batches in range(35):
            l_batches = []
            seed(time.perf_counter())
            for j in range(batch_num):
              seed(time.perf_counter())
              l_batches.append(random.randint(0,int(label.size(dim=0))-1))
            #print(l_batches)

            l_image_input = []
            l_text_input_ids = []
            l_text_input_attention = []
            l_labels = []

            for i in l_batches:
              #print(image_input[i,:])
              l_image_input.append(image_input[i,:])
              l_text_input_ids.append(text_input[i,:])
              l_labels.append(label[i])
            l_image_input = torch.stack(l_image_input)
            l_text_input_ids = torch.stack(l_text_input_ids)
            l_labels = torch.stack(l_labels)
            #print(l_image_input)
            #print(l_text_input_ids)
            #print(l_text_input_attention)
            #print(l_labels)

            
            preds = model(l_image_input, l_text_input_ids)
            #print(preds)
            #print(preds)
            #for the MSE loss function
            #loss = loss_func(preds.squeeze(), l_labels)
            #for the crossEntropy los function
            
            loss = loss_func(preds,l_labels)
            loss.backward()
            opt.step()
            opt.zero_grad()

            print('\n', f'batch #{batches}: {loss}', f'Accuracy: ({accuracy(preds,l_labels)})', end='')
            print("\n Selected batches: ", l_batches)
            print('\n', f'preds: {torch.argmax(preds, dim=1)}', f'labels: {l_labels}')
            total_loss += loss.item()
            acc += accuracy(preds,l_labels)
          
        '''
        if best_score is None:
            best_score = acc/n_batches
        else:
            # Check if val_loss improves or not.
            if acc/n_batches > best_score:
                # val_loss improves, we update the latest best_score,
                # and save the current model
                best_score = acc/n_batches
                torch.save({'state_dict':aux_model.state_dict()}, (path + "megamodel"))
                print("Best model with accuracy {} saved successfully.\n".format(acc/n_batches))
            else:
                # val_loss does not improve, we increase the counter,
                # stop training if it exceeds the amount of patience
                counter += 1
                if counter >= patience:
                    break
        '''
        sched.step(total_loss)
        print('\n', f'Epoch: ({epoch+1}/{epochs}) Loss = {total_loss/n_batches}', f'Accuracy: ({acc/n_batches})')

    torch.save(model.state_dict(), (path + "final_megamodel.pt"))    
    #torch.save(model, (path + "final_entire_megamodel.pt"))

################################################################################################################################################################################################################
####################################################################### Model's Test Functions #################################################################################################################
################################################################################################################################################################################################################
    
def test_partial(model, test_data):
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():

      image_input,text_input,target = process_data(test_data)

      n_batches = int(int(target.size(dim=0))/batch_num)
      print("\n N of batches = {}\n".format(n_batches))

      target = target.long()
      batch_start = 0
      for batches in range(int(int(target.size(dim=0))/batch_num)):
        l_image_input = []
        l_text_input_ids = []
        l_text_input_attention = []
        l_target = []
        for i in range(batch_start, (batch_start+batch_num)):
          l_image_input.append(image_input[i,:])
          l_text_input_ids.append(text_input[i,:])
          l_target.append(target[i])
        batch_start += batch_num
        l_image_input = torch.stack(l_image_input)
        l_text_input_ids = torch.stack(l_text_input_ids)
        l_target = torch.stack(l_target)

        output = model(l_image_input, l_text_input_ids)
        #pred = output.argmax(1, keepdim=True)
        test_loss = loss_func(output,l_target)
        acc = accuracy(output,l_target)
        print("\nTest set: batch: {}, Accuracy: {}; Loss: {}\n".format(batches, acc, test_loss))


def test(model, test_dl):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      image_input,text_input,target = process_data(test_dl)
      target = target.long()



      output = model(image_input, text_input['input_ids'], text_input['attention_mask'])

      #pred = output.argmax(1, keepdim=True)
      acc = accuracy(output,target)
      print('\nTest set: Accuracy: {})\n'.format(acc, len(test_dl.dataset),100. * acc))

      '''
      pred = []
      for t in output:
        pred.append(t.item())
      print(pred)
      for i in range(len(pred)):
         int_part = int(pred[i]/10)
         dbec_part = int(pred[i]%10)
         if dec_part >= 5:
            int_part+=1
         pred[i] = int_part*10
      pred = torch.tensor(pred).float()
      print(output, pred, target)
      correct += pred.eq(target.view_as(pred)).sum().item()
      test_loss /= len(test_dl.dataset)
      '''
      print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
         acc, len(test_dl.dataset),
        100. * acc))
'''
        for data, target in test_dl:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            #test_loss += f.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            print(pred, target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
'''

################################################################################################################################################################################################################
####################################################################### Training and Testing Calls #############################################################################################################
################################################################################################################################################################################################################

# Training the model and save the weights
fit(60, model, train_dl)
# to save the final model
# torch.save(model.state_dict(), "model_weights")

#model loading
#model_load = model_final(model_trans_top, trans_layer_norm, dp_rate = 0.15)
#model_load.load_state_dict(torch.load("sample_data/checkpoints/final_megamodel.pt"))
#print(model_load)

test_partial(model,train_dl)
test_partial(model,test_dl)
