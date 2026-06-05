import torch
import torch.nn as nn
from src.encoder import Encoder
from src.decoder import Decoder



class Seq2SeqTransformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model,num_heads,num_layers,max_len):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size,d_model,num_heads,num_layers,max_len)
        self.decoder = Decoder(tgt_vocab_size,d_model,num_heads,num_layers,max_len)
        
        
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        
        encoder_out ,encoder_weights = self.encoder(src,src_mask)
        
        decoder_out , decoder_weights = self.decoder(tgt,encoder_out,tgt_mask)
        
        
        return decoder_out,encoder_out
    
    
    def predict(self,encoder_input,sos_id,eos_id,max_length=50):
        
        encoder_out ,_ = self.encoder(encoder_input,None)
        
        decoder_input = torch.tensor([[sos_id]])
        
        last_token_generated = None
        
        
        
        while last_token_generated != eos_id:
            
            causal_mask = (torch.tril(torch.ones(decoder_input.size(1),decoder_input.size(1))) == 0)
            
            prediction,_ = self.decoder(decoder_input,encoder_out,causal_mask)
            
            
            print(f"This is the shape of prediction: {prediction.shape}")
            
            latest_prediction = prediction[:,-1,:]
            
            print(f"This is the shape of latest prediction: {latest_prediction.shape}")
            
            _,next_word_id = torch.max(latest_prediction,dim=1)
            
            print(f"This is the shape of next_word_id: {next_word_id.shape}")
            next_word_id = next_word_id.unsqueeze(1)
            
            print(f"This is the shape of unsqueezed next_word_id: {next_word_id.shape}")

            
            decoder_input = torch.cat([decoder_input,next_word_id],dim=1)
            
            print(f"This is the shape of decoder_input: {decoder_input.shape}")
            
            last_token_generated = next_word_id.item() 
            
            print(f"This is the shape of last_token_generated: {last_token_generated}")  
            
                     
            
            
            
        final_ids = decoder_input.squeeze().cpu().tolist()
        
        
        return final_ids
            
            
        
        