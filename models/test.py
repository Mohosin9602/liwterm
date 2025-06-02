import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from torchvision import transforms
from torch.utils.data import DataLoader
from utils import process_data, process_data_2, accuracy

################################################################################################################################################################################################################
####################################################################### Confusion Matrix #######################################################################################################################
################################################################################################################################################################################################################

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    
    # Handle row and column tick marks separately
    x_tick_marks = np.arange(len(df_confusion.columns))
    y_tick_marks = np.arange(len(df_confusion.index))
    
    plt.xticks(x_tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(y_tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.show()

################################################################################################################################################################################################################
####################################################################### Model's Test Functions #################################################################################################################
################################################################################################################################################################################################################
    
def test_partial(model, test_data, batch_num, model_config):
    #_COVERSION = {"": ""}
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = nn.CrossEntropyLoss()
    out_labels = []
    out_preds = []
    
    # Get device from model
    device = next(model.parameters()).device #GPUUUU
    
    with torch.no_grad():

      image_input,text_input,target = process_data(test_data, model_config)
      
      # Move tensors to device
      image_input = image_input.to(device) #GPUUUU
      text_input = text_input.to(device) #GPUUUU
      target = target.to(device) #GPUUUU
      
      overall_acc = 0
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
        
        # Move batch tensors to device
        l_image_input = l_image_input.to(device) #GPUUUU
        l_text_input_ids = l_text_input_ids.to(device) #GPUUUU
        l_target = l_target.to(device) #GPUUUU
        
        starting_time = time.time()
        output = model(l_image_input, l_text_input_ids.to(torch.float32), model_config)
        #pred = output.argmax(1, keepdim=True)
        test_loss = loss_func(output,l_target)
        acc = accuracy(output,l_target)
        overall_acc += float(acc)
        #print(torch.argmax(output, dim=1).squeeze().tolist())
        #print(l_target.squeeze().tolist())
        out_preds += torch.argmax(output, dim=1).squeeze().tolist()
        out_labels += l_target.squeeze().tolist()
        print("\nTest set: batch: {}, Accuracy: {}; Loss: {}\n".format(batches, acc, test_loss))
      #print("Preds: ", len(out_preds))
      #print("Labels: ", len(out_labels))
      overall_acc/= int(int(target.size(dim=0))/batch_num)
      print("Overall Accuracy: ", overall_acc)
      print("Test Labels: ", out_labels)
      print("Test Predictions: ", out_preds)
      print("Inference Time: ", time.time() - starting_time)
      # Standard confusion matrix format: true labels as rows, predictions as columns
      df_confusion = pd.crosstab(out_labels, out_preds, rownames=['Actual'], colnames=['Predicted'])
      print(df_confusion)
      plot_confusion_matrix(df_confusion)


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

