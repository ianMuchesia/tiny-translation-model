import string
import json
class Tokenizer:
    def __init__(self):
        # Control tokens
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        self.EOS = "<EOS>"
        self.SOS = "<SOS>"
        
        # Single vocab mapping
        self.word2int = {
            self.PAD: 0,
            self.UNK: 1,
            self.EOS: 2,
            self.SOS: 3 
        }
        
        self.int2word = {
            0: self.PAD,
            1: self.UNK,
            2: self.EOS,
            3: self.SOS
        }
        
    def clean_text(self, text: str) -> str:
        text = text.lower()
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)
    
    def build_vocab(self, corpus: str):
        clean = self.clean_text(corpus)
        unique_words = set(clean.split())
        
        for word in unique_words:
            if word not in self.word2int:
                new_id = len(self.word2int)
                self.word2int[word] = new_id
                self.int2word[new_id] = word
                
    def encode(self, sentence: str) -> list:
        # Start with SOS
        encoded = [self.word2int[self.SOS]]
        
        clean = self.clean_text(sentence)
        for word in clean.split():
            # Lookup word, fallback to UNK ID if missing
            word_id = self.word2int.get(word, self.word2int[self.UNK])
            encoded.append(word_id)
            
        # End with EOS
        encoded.append(self.word2int[self.EOS])
        return encoded
    
    def decode(self,word_ids:list) ->str:
        
        words = []
        
        for word_id in word_ids:
            word = self.int2word.get(word_id,self.UNK)
            
            words.append(word)
            
            
        return " ".join(words)
    
    
    def save(self,filepath):
        with open(filepath,"w") as f:
            json.dump(self.word2int,f)
            
            
        
    def load(self,filepath):
        with open(filepath,"r") as f:
            self.word2int = json.load(f)
            
            self.int2word = {idx: word for word, idx in self.word2int.items()}
            
        
    
    