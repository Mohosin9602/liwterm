# Transfuse 

## Introduction
This is the Transfuse repository in which a ML-muultimodal model is designed for the detection of 6 skin lesions, including melanoma. The purpose of such a model is to combine both images and text (from anamnese to enhance the correct classification of skin diseases. The main technique employed here is the `Transformers` for images and text feature calculations, i.e., using `Vision Transformers` and Bert Text Processing, respectively. Afterwards, a light-weighted neural networks model is fed with such features, combine them and provide the final skin lesion classification.  

## Setting up your Environment
To make sure you have all the correct packages and libs to work, please run the `requirements` file in a fresh conda environment.

## Setting up your Dataset
