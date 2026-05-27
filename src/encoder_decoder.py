
import torch.nn as nn
from src.encoder import Encoder
from src.decoder import Decoder



class Seq2SeqTransformer(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,num_layers,max_len):
        super().__init__()
        
        self.encoder = Encoder(vocab_size,d_model,num_heads,num_layers,max_len)
        self.decoder = Decoder(vocab_size,d_model,num_heads,num_layers,max_len)
        
        
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        
        encoder_out ,encoder_weights = self.encoder(src,src_mask)
        
        decoder_out , decoder_weights = self.decoder(tgt,encoder_out,tgt_mask)
        
        
        return decoder_out