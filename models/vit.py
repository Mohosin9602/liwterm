from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTModel, ViTConfig
import torch.nn as nn
import torch.nn.functional as F

################################################################################################################################################################################################################
####################################################################### Image Model Definitions ################################################################################################################
################################################################################################################################################################################################################

def vit_model(pre_trained_link):
    """
    Initialize and configure a Vision Transformer model.
    
    Args:
        pre_trained_link: name or path of the pre-trained ViT model
    
    Returns:
        tuple: (model_trans_top, trans_layer_norm) - configured ViT model parts
    """
    # Initialize the ViT model
    model_trans = ViTModel.from_pretrained(pre_trained_link)
    
    # Freeze early layers
    count = 0
    for child in model_trans.children():
        count += 1
        if count < 4:
            for param in child.parameters():
                param.requires_grad = False

    # Get all the layers from the Transformer model
    layers_trans = list(model_trans.children())
    
    # Remove the normalization layer and pooler layer
    model_trans_top = nn.Sequential(*layers_trans[:-2])
    
    # Get the normalization layer
    trans_layer_norm = list(model_trans.children())[2]
    
    return (model_trans_top, trans_layer_norm)

