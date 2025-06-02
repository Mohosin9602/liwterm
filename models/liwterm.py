import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time

# Size of text feature vector
TOKEN_SIZE = 768

################################################################################################################################################################################################################
####################################################################### Transfuse Definitions ##################################################################################################################
################################################################################################################################################################################################################

class model_final(nn.Module):
    """
    Final model that combines vision and text features for classification.
    
    Args:
        model_trans_top: top layers of the vision transformer
        trans_layer_norm: normalization layer from vision transformer
        n_classes: number of output classes
        dp_rate: dropout rate (default: 0.3)
    """
    def __init__(self, model_trans_top, trans_layer_norm, n_classes, dp_rate=0.3):
        super().__init__()
        # Vision transformer layers
        self.model_trans_top = model_trans_top
        self.trans_layer_norm = trans_layer_norm
        self.trans_flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.trans_linear = nn.Linear(150528, 4096)
        
        # Batch normalization layers
        self.batchnorm0 = nn.BatchNorm1d(4864)
        self.batchnorm1 = nn.BatchNorm1d(4096)
        self.batchnorm2 = nn.BatchNorm1d(2048)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.batchnorm6 = nn.BatchNorm1d(768)
        
        # Text processing layers
        self.token_flatten = nn.Flatten()
        self.token_layer_norm = nn.LayerNorm(TOKEN_SIZE)
        self.token_linear = nn.Linear(TOKEN_SIZE, 2048)
        
        # Shared layers
        self.dropout = nn.Dropout(dp_rate)
        self.linear_complete = nn.Linear(4864, 2048)
        self.linear_words = nn.Linear(768, 1024)
        self.linear_vit = nn.Linear(4096, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, trans_b, token_b_input_feats, model_config):
        """
        Forward pass of the model.
        
        Args:
            trans_b: batch of image features
            token_b_input_feats: batch of text features
            model_config: configuration string ('words', 'ViT', or 'complete')
            
        Returns:
            torch.Tensor: logits for each class
        """
        # Process vision features
        result_trans = self.model_trans_top(trans_b)
        patch_state = result_trans.last_hidden_state[:,1:,:] # Remove classification token
        result_trans = self.trans_layer_norm(patch_state)
        result_trans = self.trans_flatten(patch_state)
        result_trans = self.dropout(result_trans)
        result_trans = self.trans_linear(result_trans)
        result_trans = self.dropout(result_trans)

        # Process text features
        result_token = self.token_flatten(token_b_input_feats)
        result_token = self.token_layer_norm(result_token)
        
        # Process based on model configuration
        if model_config == "words":
            # Text-only path
            result = result_token
            result = self.dropout(result)
            result = self.relu(result)
            result = self.batchnorm6(result)
            
            result = self.linear_words(result)
            result = self.batchnorm3(result)
            result = self.dropout(result)
            result = self.relu(result)
            
        elif model_config == "ViT":
            # Vision-only path
            result = result_trans
            result = self.dropout(result)
            result = self.relu(result)
            result = self.batchnorm1(result)
            
            result = self.linear_vit(result)
            result = self.batchnorm2(result)
            result = self.dropout(result)
            result = self.relu(result)
            
            result = self.linear2(result)
            result = self.batchnorm3(result)
            result = self.dropout(result)
            result = self.relu(result)
            
        else:
            # Combined vision and text path
            result = torch.cat((result_trans, result_token), 1)
            result = self.dropout(result)
            result = self.relu(result)
            result = self.batchnorm0(result)
            
            result = self.linear_complete(result)
            result = self.batchnorm2(result)
            result = self.dropout(result)
            result = self.relu(result)
            
            result = self.linear2(result)
            result = self.batchnorm3(result)
            result = self.dropout(result)
            result = self.relu(result)
        
        # Final layers (shared across all paths)
        result = self.linear3(result)
        result = self.batchnorm4(result)
        result = self.relu(result)
        
        result = self.linear4(result)
        result = self.batchnorm5(result)
        result = self.relu(result)
        
        result = self.linear5(result)
        
        return result  # Return logits (cross entropy loss expects logits)

