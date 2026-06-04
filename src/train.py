
import torch 
import torch.nn as nn

def train_step(model,batch,optimizer,criterion,device,running_loss,total,correct):
    
    
    encoder_input,decoder_input, expected_labels = batch
    
    # Push to GPU/CPU
    encoder_input = encoder_input.to(device)
    decoder_input = decoder_input.to(device)
    expected_labels = expected_labels.to(device)
    
    optimizer.zero_grad()


   
    
    seq_len = decoder_input.size(1)
    src_mask =(encoder_input == 0).unsqueeze(1).unsqueeze(2)
    
    
    
    causal_mask = (torch.tril(torch.ones(seq_len,seq_len)) == 0)
    
    
    decoder_padding_mask = (decoder_input == 0).unsqueeze(1).unsqueeze(2)
    
   
    
    tgt_mask = decoder_padding_mask | causal_mask
    
    
   
    
    #Forward pass, the model only see the decoder input
    decoder_out,encoder_out = model(encoder_input,decoder_input,src_mask,tgt_mask)
    
    
    
    # Flatten predictions to [Batch * Seq_Len, Vocab_Size] -> [160, 1000]
    flat_decoder_out = decoder_out.view(-1, decoder_out.size(-1))

    # Flatten labels to [Batch * Seq_Len] -> [160]
    flat_expected_labels = expected_labels.view(-1)
    
    
    #Calculate the error(Loss) between predictions and expected_labels
    loss = criterion(flat_decoder_out,flat_expected_labels)
    
    loss.backward()
    
    
    optimizer.step()
    
    
    running_loss += loss.item()
    
    
    print(f"The loss is : {loss.item()}")
    
    _,predicted = torch.max(decoder_out.data,2)
    
    # print(f"This is the predicted: {predicted}")
    
    
    total += expected_labels.size(0)
    
    print(f"the shape of predicted is {predicted.shape}")
    print(f"the shape of expected_labels is  {expected_labels.shape}")
    correct += (predicted == expected_labels).sum().item()
    
    
    return total,correct,running_loss
        
    
    
    
     
    