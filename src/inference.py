import torch
from src.greedy_decode import GreedyDecoder


class Translator:
    def __init__(self,model,eng_tokenizer,swa_tokenizer,sos_id,eos_id):
        self.eng_tokenizer = eng_tokenizer
        self.swa_tokenizer = swa_tokenizer
        self.model = model
        self.sos_id = sos_id
        self.eos_id = eos_id
    
    def preprocess(self,input):
        
        encoded_input = self.eng_tokenizer.encode(input)
        
        tensor_input = torch.tensor([encoded_input])
        
        return tensor_input
    
    def encode_input(self,input):
        
        tensor_input = self.preprocess(input)
        
        
        
        encoder_out ,_ = self.model.encoder(tensor_input,mask=None)
        
        
        return encoder_out
    
    def generate_translation(self, encoder_out):
        # 1. Spin up the decoding algorithm and hand it the model
        decoder = GreedyDecoder(self.model, self.sos_id, self.eos_id)
        
        # 2. Run the while loop to get the list of integers
        raw_ids = decoder.decode(encoder_out)
        
        return raw_ids
    
    def postprocess(self,encoded_response):
        
        return self.swa_tokenizer.decode(encoded_response[1:-1])
    
    
    def translate(self,text):
        
        encoder_out = self.encode_input(text)
        
        raw_ids =self.generate_translation(encoder_out)
        
        output = self.postprocess(raw_ids)
        
        return output
    
    
    
        
        
        
        