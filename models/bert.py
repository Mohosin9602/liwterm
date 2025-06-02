from transformers import ViTFeatureExtractor, ViTImageProcessor, ViTModel, ViTConfig, AutoConfig, AutoTokenizer, AutoModel, BertModel, AutoModelForSequenceClassification, pipeline

################################################################################################################################################################################################################
####################################################################### Text Model Definitions #################################################################################################################
################################################################################################################################################################################################################

def bert_model(model_name='bert-base-uncased'):
    """
    Initialize and configure a BERT model for text feature extraction.
    
    Args:
        model_name: name of the pre-trained BERT model to use
    
    Returns:
        model: configured BERT model with frozen layers
    """
    # Initialize the BERT model
    model_token = BertModel.from_pretrained(model_name)
    
    # Freeze early layers
    count = 0
    for child in model_token.children():
        count += 1
        if count < 4:
            for param in child.parameters():
                param.requires_grad = False
    
    return model_token

