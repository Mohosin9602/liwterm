import sys
sys.path.append('../')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random

from random import seed
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import process_data, process_data_2, accuracy


################################################################################################################################################################################################################
####################################################################### Model's Fit Function ###################################################################################################################
################################################################################################################################################################################################################

def fit(epochs, model, train_dl, optimizer, lr_scheduler, batch_num, dataset_name, model_config):
    opt = optimizer
    sched = lr_scheduler
    loss_func = nn.CrossEntropyLoss()
    best_score = None
    patience = 50
    path = "sample_data/checkpoints/" # user_defined path to save model
    
    print("Calculating the features...")
    image_input,text_input,label = process_data(train_dl, dataset_name)
    
    print("Feature sizes: ViT({}); pipeline({}); labels({}).".format(image_input.size(), text_input.size(), label.size()))
    
    if not os.path.exists(path):
      os.makedirs(path)
    print("Training...\n")
    acc_training = []
    loss_training = []
    for epoch in range(epochs):
        
        counter = 0
        model.train()
       
        acc = 0
        total_loss = 0
        
        label = label.long()
        n_batches = int(int(label.size(dim=0))/batch_num)
        penalty = 0.001

        #number of batches
        for batches in range(int(int(label.size(dim=0))/batch_num)):
        #for batches in range(2):    
            l_batches = []
            seed(time.perf_counter())
            for j in range(batch_num):
              seed(time.perf_counter())
              l_batches.append(random.randint(0,int(label.size(dim=0))-1))

            l_image_input = []
            l_text_input_ids = []
            l_text_input_attention = []
            l_labels = []

            for i in l_batches:
              l_image_input.append(image_input[i,:])
              l_text_input_ids.append(text_input[i,:])
              l_labels.append(label[i])
            l_image_input = torch.stack(l_image_input)
            l_text_input_ids = torch.stack(l_text_input_ids)
            l_labels = torch.stack(l_labels)

            
            preds = model(l_image_input, l_text_input_ids.to(torch.float32), model_config)
            loss = loss_func(preds,l_labels)

            out_preds = torch.argmax(preds, dim=1).squeeze().tolist()
            out_labels = l_labels.squeeze().tolist()
            '''
            #verifying if the gradient punishing is required (only for class 5 misclassification)
            dg_punish = 0
            for i in range(len(out_labels)):
               if out_labels[i] == 5 and out_preds[i] != 5:
                  dg_punish = 1
                  break

            # This is where the activation gradients are computed
            # But it makes clear that we're *only* interested in the activation gradients at this point
            if dg_punish:
              grads = torch.autograd.grad(loss, [model.relu1, model.relu2, model.relu3, model.relu4, model.relu5, model.softmaxact], create_graph=True, only_inputs=True)
              grad_norm = 0
              for grad in grads:
                #L2 penalty
                grad_norm += grad.pow(2).sum()
              print("grad_nomr: ",grad_norm)
              #TODO check if class 5 was mispredicted, and then apply the gradient penalty
              loss = loss + grad_norm * penalty
            '''
            loss.backward()
            opt.step()
            opt.zero_grad()

            print('\n', f'batch #{batches}: {loss}', f'Accuracy: ({accuracy(preds,l_labels)})', end='')
            print("\n Selected batches: ", l_batches)
            print('\n', f'preds: {torch.argmax(preds, dim=1)}', f'labels: {l_labels}')
            total_loss += loss.item()
            acc += accuracy(preds,l_labels)
          
       
        sched.step(total_loss)
        print('\n', f'Epoch: ({epoch+1}/{epochs}) Loss = {total_loss/n_batches}', f'Accuracy: ({acc/n_batches})')
        acc_training.append(acc/n_batches)
        loss_training.append(total_loss/n_batches)

    torch.save(model.state_dict(), (path + "final_megamodel.pt"))    
    with open("loss_training.txt", "w") as f:
       f.write('\n'.join(str(loss_values) for loss_values in loss_training))

    with open("acc_training.txt", "w") as f:
       f.write('\n'.join(str(acc_values.item()) for acc_values in acc_training))
