# Transfuse 

## :mortar_board: Introduction
This is the Transfuse repository in which a ML-multimodal model is designed for the detection of 6 skin lesions, including melanoma. The purpose of such a model is to combine both images and text (from anamnese to enhance the correct classification of skin diseases. The main technique employed here is the `Transformers` for images and text feature calculations, i.e., using `Vision Transformers` and Bert Text Processing, respectively. Afterwards, a light-weighted neural networks model is fed with such features, combine them and provide the final skin lesion classification.  

## :school_satchel: Setting up your Environment
To make sure you have all the correct packages and libs to work, please run the `requirements` file in a fresh conda environment.

## :floppy_disk: Setting up your Dataset
Currently, you have to extract all the files from [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) and store them inside `data/imgs_1_2_3`. The metadata file and the splits (training and test) must be in the folder `data/`. By now, a basic split and the metadata files are available, but you are encouraged to change the split protocol if you want to.

## :trollface: Relevant Information
Right now, the `__getitem__` function from the `customDataloader` class is unavailable. Due to the intense and computational-cost processes of computing ViT and text (Bert) features, this function presented unexpected behavior, harming its use. Its implementations is under progress.

Hence, the features are calculated for the entire training and test sets and the batches for each training-epoch are manually calculated (please check the `fit` function inside `models/training.py`). Once the `__getitem__` implementation is finished, the use of anautomatic batch selection from `DataLoader` class will be employed, updating the current batch selection.

After setting up the environment and storing (properly) the data, all you have to do is run the `main.py` file followed by the selected dataset (`padufes20` or `isic19`) and the model configuration (`ViT`, `words`, or `complete).

Finally, a folder named `sample_data/checkpoints` will be created with the final model inside, after the training process is finished. You can load the model for further inference. 
