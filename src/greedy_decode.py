import torch



class GreedyDecoder:
    def  __init__(self,model,sos_id,eos_id,max_length=50):
        self.model = model
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.max_length = max_length
        
        
    
    def stop_condition(self, current_token, current_length):
        return current_token == self.eos_id or current_length == self.max_length
    
    
    def predict_next_token(self,decoder_input,encoder_out):
        
        
        causal_mask = (torch.tril(torch.ones(decoder_input.size(1),decoder_input.size(1))) == 0)
        
        prediction,decoder_weights= self.model.decoder(decoder_input,encoder_out,causal_mask)
        
        
        
        latest_prediction = prediction[:,-1,:]
        
        
        _,next_word_id = torch.max(latest_prediction,dim=1)
        
        next_word_id = next_word_id.unsqueeze(1)
        
       

        
        decoder_input = torch.cat([decoder_input,next_word_id],dim=1)
        
        print(f"This is the shape of decoder_input: {decoder_input.shape}")
        
        last_token_generated = next_word_id.item() 
        
        print(f"This is the shape of last_token_generated: {last_token_generated}")  
        
        return decoder_input,last_token_generated
        
    def decode(self,encoder_out):
        
        decoder_input = torch.tensor([[self.sos_id]])
        
        current_token = None
        
       
        while not self.stop_condition(current_token,decoder_input.size(-1)):
            decoder_input,current_token = self.predict_next_token(decoder_input,encoder_out)
            
            
        final_ids = decoder_input.squeeze().cpu().tolist()
        
        
        return final_ids
            
        