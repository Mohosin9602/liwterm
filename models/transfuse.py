import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time


################################################################################################################################################################################################################
####################################################################### Transfuse Definitions ##################################################################################################################
################################################################################################################################################################################################################

# Merge the two models
class model_final(nn.Module):
    def __init__(self, model_trans_top, trans_layer_norm, n_classes, dp_rate = 0.3):
        super().__init__()
        # All the trans model layers
        self.model_trans_top = model_trans_top
        self.trans_layer_norm = trans_layer_norm
        self.trans_flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.trans_linear = nn.Linear(150528, 4096)
        self.batchnorm0 = nn.BatchNorm1d(4864)
        self.batchnorm1 = nn.BatchNorm1d(4096)
        self.batchnorm2 = nn.BatchNorm1d(2048)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.batchnorm4 = nn.BatchNorm1d(512)
        
        # All the text model
        #by now, only the features are being added
        self.token_flatten = nn.Flatten()
        self.token_layer_norm = nn.LayerNorm(768)
        self.token_linear = nn.Linear(768, 1024)
        
        #to incorporate bert model
        '''
        self.model_token_top = model_token_top
        self.token_flatten = nn.Flatten()
        self.token_linear = nn.Linear(768, 2048)
        '''

        # Merge the result and pass the
        self.dropout = nn.Dropout(dp_rate)
        #self.linear0 = nn.Linear(5960, 4096)
        self.linear1 = nn.Linear(4864, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
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

        # Get text features
        result_token = self.token_flatten(token_b_input_feats)
        result_token = self.token_layer_norm(result_token)
        #result_token = self.dropout(result_token)
        #result_token = self.token_linear(result_token)
        
        #To incorporate features from bert model
        '''
         # Get intermediate outputs using hidden layer
        result_token = self.model_token_top(token_b_input_ids, attention_mask=token_b_attention_masks)
        patch_state = result_token.last_hidden_state[:,1,:] # Remove the classification token and get the last hidden state of all patchs
        result_token = self.token_flatten(patch_state)
        result_token = self.dropout(result_token)
        result_token = self.token_linear(result_token)
        '''

        result_merge = torch.cat((result_trans, result_token),1)
        result_merge = self.dropout(result_merge)
        result_merge = self.relu(result_merge)
        result_merge = self.batchnorm0(result_merge)

        #result_merge = self.linear0(result_merge)
        #result_merge = self.batchnorm1(result_merge)
        #result_merge = self.dropout(result_merge)
        #result_merge = self.relu(result_merge)
        
        result_merge = self.linear1(result_merge)
        result_merge = self.batchnorm2(result_merge)
        result_merge = self.dropout(result_merge)
        result_merge = self.relu(result_merge)
        
        result_merge = self.linear2(result_merge)
        result_merge = self.batchnorm3(result_merge)
        result_merge = self.dropout(result_merge)
        result_merge = self.relu(result_merge)
        
        result_merge = self.linear3(result_merge)
        result_merge = self.batchnorm4(result_merge)
        result_merge = self.relu(result_merge)
        
        result_merge = self.linear4(result_merge)        

        return result_merge

