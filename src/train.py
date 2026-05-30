
import torch 
import torch.nn as nn

def train_step(model,src,tgt,src_mask, tgt_mask,optimizer,criterion):
    
    optimizer.zero_grad()
    
    
    #2. Slice the targets (Teacher Forcing)
    decoder_input = tgt[:,:-1]
    
    expected_labels = tgt[:,1:]
    
    
    #we must also slice the tgt_mask to match the new lenght of 9!
    sliced_tgt_mask = tgt_mask[:,:-1,:-1]
    
    #Forward pass, the model only see the decoder input
    decoder_out,encoder_out = model(src,decoder_input,src_mask,sliced_tgt_mask)
    
    
    #Calculate the error(Loss) between predictions and expected_labels
    loss = criterion(decoder_out,expected_labels)
    
    loss.backward()
    
    
    optimizer.step()
    
    
    print(f"The loss is : {loss.item()}")
    
    _,predicted = torch.max(decoder_out.data,1)
    
    print(f"This is the predicted: {predicted}")
    
    
    
     
    