
import torch.nn as nn
from src.attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff=2048 ):
        super().__init__()
        
        
        self.mha = MultiHeadAttention(d_model,num_heads)
        self.cha = MultiHeadAttention(d_model,num_heads)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        
    
    def forward(self,x,encoder_out,tgt_mask):
        
        # --- MULTI-HEAD ATTENTION BLOCK ---
        # 1. Normalize a COPY of the input for the heavy machinery
        norm_x = self.norm1(x)
        
        # 2. Run the normalized data through the attention heads
        attn_out, w = self.mha(norm_x, norm_x,norm_x,mask=tgt_mask)
        
        
        out1 = attn_out + x
        
        norm_out1 = self.norm2(out1)
        
       
        
        attn_out2 ,w2 = self.cha(norm_out1,encoder_out,encoder_out,None)
        
        
        out2 = attn_out2 + out1
        
        
        norm_out2 = self.norm3(out2)
        
        
        ffn_out = self.ffn(norm_out2)
        
        
        out3 = ffn_out + out2
        
        return out3 , w2
        
        
        
