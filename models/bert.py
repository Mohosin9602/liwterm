################################################################################################################################################################################################################
####################################################################### Text Model Definitions #################################################################################################################
################################################################################################################################################################################################################

# Modifying the tokenizer model - metadata
def bert_model():
	count = 0
	for child in model_token.children():
	    #print(child, count)
	    count += 1
	    if count < 4:
		for param in child.parameters():
		    param.requires_grad = False

	layers_token = list(model_token.children()) # Get all the layers from the Transformer model

	#model_token_top = deleteEncodingLayers(model_token, list(layers_token[:-1])) # Remove the pooler layer
	model_token_top = model_token
	return model_token_top

