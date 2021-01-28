import torch
import numpy as np
import torch.nn.functional as F

class Batch:

    def __init__(self, src, trg, out_pad_idx, device):
        
        self.src = src
        
        self.trg = None
        self.trg_y = None
        self.ntokens = None

        if trg is not None:
            self.trg = trg[:-1,]
            self.trg_y = trg[1:].reshape(-1)
            self.ntokens = (self.trg_y != out_pad_idx).sum().item()  
        
        if device == torch.device('cuda'):
            self.src = self.src.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                

def rebatch(batch, out_pad_idx, device):
    return Batch(batch.src, batch.trg, out_pad_idx, device)