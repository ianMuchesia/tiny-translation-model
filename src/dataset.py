from torch.utils.data import Dataset
from src.tokenizer import Tokenizer



## The DataLoader will automatically call __getitem__ 32 times  in training loop

class TranslationDataset(Dataset):
    def __init__(self, filepath, eng_tokenizer, swa_tokenizer):
        self.filepath = filepath
        self.eng_tokenizer = eng_tokenizer
        self.swa_tokenizer = swa_tokenizer
        
        self.eng_sentences = []
        self.swa_sentences = []
        
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                # split the line and strip spaces from the pieces
                sentence = line.strip().split("=>")
                self.eng_sentences.append(sentence[0].strip())
                self.swa_sentences.append(sentence[1].strip())
                
                
        
        # Build the vocabularies cleanly from the extracted text
        # We join the entire list into one massive string and feed it to your method
        self.eng_tokenizer.build_vocab(" ".join(self.eng_sentences))
        self.swa_tokenizer.build_vocab(" ".join(self.swa_sentences))
                
    def __len__(self):
        
        return len(self.eng_sentences)
        
        
    def __getitem__(self, index):
        
        eng_text = self.eng_sentences[index]
        
        swa_text = self.swa_sentences[index]
        
        
        eng_encoded = self.eng_tokenizer.encode(eng_text)
        swa_encoded = self.swa_tokenizer.encode(swa_text)
        
        
        #setup the inputs and targets
        encoder_input = eng_encoded
        decoder_input = swa_encoded[:-1]
        decoder_target = swa_encoded[1:]        
        
        return encoder_input, decoder_input, decoder_target
    
            
            
            
            
            
            
        
        
        