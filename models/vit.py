################################################################################################################################################################################################################
####################################################################### Image Model Definitions ################################################################################################################
################################################################################################################################################################################################################

def vit_model():
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
	
	return (model_trans_top, trans_layer_norm)

