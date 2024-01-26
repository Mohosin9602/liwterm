################################################################################################################################################################################################################
####################################################################### Model's Fit Function ###################################################################################################################
################################################################################################################################################################################################################

def fit(epochs, model, train_dl, optimizer, lr_scheduler):
    opt = optimizer
    sched = lr_scheduler
    loss_func = nn.CrossEntropyLoss()
    best_score = None
    patience = 50
    path = "sample_data/checkpoints/" # user_defined path to save model
    
    print("Calculating the features...")
    image_input,text_input,label = process_data(train_dl)
    
    print("Feature sizes: ViT({}); pipeline({}); labels({}).".format(image_input.size(), text_input.size(), label.size()))
    
    if not os.path.exists(path):
      os.makedirs(path)
    print("Training...\n")
    for epoch in range(epochs):
        
        counter = 0
        model.train()
       
        acc = 0
        total_loss = 0
        
        label = label.long()
        n_batches = int(int(label.size(dim=0))/batch_num)
        
        #number of batches
        for batches in range(int(int(label.size(dim=0))/batch_num)):

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

            
            preds = model(l_image_input, l_text_input_ids)
            
            loss = loss_func(preds,l_labels)
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

    torch.save(model.state_dict(), (path + "final_megamodel.pt"))    

