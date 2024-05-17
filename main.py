# -*- coding: utf-8 -*-
"""TransFuse.ipynb
Luis Souza
la.souza@inf.ufes.br
"""

#imports

import os
import pandas as pd
import numpy as np
import torch

from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTModel, ViTConfig
from torch.utils.data import DataLoader
from datasets import Dataset
from utils import process_metadata_frame, customDataset, process_data, set_params
from models.vit import vit_model
from models.bert import bert_model
from models.transfuse import model_final
from models.train import fit
from models.test import test_partial

#checking the current folder
print(os.getcwd())

#batch size definition
batch_size = 24

#n_classes
n_classes = 6
folder = 1

#ViT Feature Transformation version
trans_version = 'google/vit-large-patch16-224'
vit_weights_version = 'google/vit-base-patch16-224-in21k'

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

#folder filtering
#TODO use only train folders - validation file is only for testing (folder == 6)
df = df.loc[df["folder"] == folder]
df_test = df_test.loc[df_test["folder"] == folder]
df = df.drop("folder", axis=1)
df_test = df_test.drop("folder", axis=1)

classes = tuple(df["diagnostics_class"].unique())
print(classes)

print(df)
print(len(df))

#Loaders definition
#This transformation is required for the data loading and dataloader creation
trans_transform = ViTFeatureExtractor.from_pretrained(trans_version)

train_ds = customDataset(df, trans_transform=trans_transform)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

test_ds = customDataset(df_test, trans_transform=trans_transform)
test_dl = DataLoader(test_ds, batch_size=16, shuffle=True)

print(test_dl.dataset.labels)

#ViT model
model_trans_top, trans_layer_norm = vit_model(vit_weights_version)
print(model_trans_top)

#Transfuse model
model = model_final(model_trans_top, trans_layer_norm, n_classes, dp_rate = 0.3)
# model.load_state_dict(torch.load('model_weights_1228'))

print(model)
 
# Define optimizer and learning_rate scheduler
optimizer, lr_scheduler = set_params(model)

#Training the model and save the weights
fit(1, model, train_dl, optimizer, lr_scheduler, batch_size)

#for loading the saved model model loading
#model_load = model_final(model_trans_top, trans_layer_norm, dp_rate = 0.15)
#model_load.load_state_dict(torch.load("sample_data/checkpoints/final_megamodel.pt"))
#print(model_load)

#Testing
test_partial(model,test_dl, batch_size)
