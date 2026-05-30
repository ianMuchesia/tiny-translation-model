import torch
from torch.nn.utils.rnn import pad_sequence

def dynamic_pad_collate(batch):
    # 1. Rotate the matrix to separate English, Swahili Input, and Swahili Target
    eng_list, swa_in_list, swa_tgt_list = zip(*batch)
    
    # 2. THE FIX: Cast the raw Python lists into PyTorch Tensors
    eng_tensors = [torch.tensor(seq, dtype=torch.long) for seq in eng_list]
    swa_in_tensors = [torch.tensor(seq, dtype=torch.long) for seq in swa_in_list]
    swa_tgt_tensors = [torch.tensor(seq, dtype=torch.long) for seq in swa_tgt_list]
    
    # 3. Pass the valid Tensors into the C++ padding function
    eng_padded = pad_sequence(eng_tensors, batch_first=True, padding_value=0)
    swa_in_padded = pad_sequence(swa_in_tensors, batch_first=True, padding_value=0)
    swa_tgt_padded = pad_sequence(swa_tgt_tensors, batch_first=True, padding_value=0)
    
    return eng_padded, swa_in_padded, swa_tgt_padded