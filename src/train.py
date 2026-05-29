


def train_step(model,src,tgt,src_mask, tgt_mask,optimizer,criterion):
    
    optimizer.zero_grad()
    
    
    #2. Slice the targets (Teacher Forcing)
    decoder_input = tgt[:,:-1]
    
    expected_labels = tgt[:,1:]
    
    
    #we must also slice the tgt_mask to match the new lenght of 9!
    sliced_tgt_mask = tgt_mask[:,:-1,:-1]
    
    #Forward pass, the model only see the decoder input
    predictions = model(src,decoder_input,src_mask,sliced_tgt_mask)
    
    
    #Calculate the error(Loss) between predictions and expected_labels
    loss = criterion(predictions,expected_labels)
     
    