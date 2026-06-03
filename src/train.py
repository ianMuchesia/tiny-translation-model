
import torch 
import torch.nn as nn

def train_step(model,batch,optimizer,criterion,device):
    
    
    encoder_input,decoder_input, expected_labels = batch
    
    # Push to GPU/CPU
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    expected_labels = expected_labels.to(device)
    
    optimizer.zero_grad()


    print(f"this is the shape of encoder_input {encoder_input.shape}")
    print(f"this is the shape of decoder_input {decoder_input.shape}")
    
    seq_len = decoder_input.size(1)
    src_mask =(encoder_input == 0)
    
    
    causal_mask = (torch.tril(torch.ones(seq_len,seq_len)) == 0)
    
    
    decoder_padding_mask = (decoder_input == 0).unsqueeze(1)
    
    tgt_mask = decoder_padding_mask | causal_mask
    
    
   
    
    #Forward pass, the model only see the decoder input
    decoder_out,encoder_out = model(encoder_input,decoder_input,src_mask,tgt_mask)
    
    print(f"this is the shape of decoder_out {decoder_out.shape}")
    
    print(f"this is the shape of encoder_out {encoder_out.shape}")
    
    
    print(f"this is the shape of expected_labels {expected_labels.shape}")
    
    
    
    #Calculate the error(Loss) between predictions and expected_labels
    loss = criterion(decoder_out,expected_labels)
    
    loss.backward()
    
    
    optimizer.step()
    
    
    print(f"The loss is : {loss.item()}")
    
    _,predicted = torch.max(decoder_out.data,1)
    
    print(f"This is the predicted: {predicted}")
    
    
    
     
    