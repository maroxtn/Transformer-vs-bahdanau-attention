import torch

class Batch:

    def __init__(self, src, trg, device, in_pad_idx, out_pad_idx):
        
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src == in_pad_idx)
        
        self.trg = None
        self.trg_y = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg
            self.trg_lengths = trg_lengths
            self.trg_y = trg[1:].reshape(-1)
            self.ntokens = (self.trg_y != out_pad_idx).sum().item()  
        
        if device == torch.device('cuda'):
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                


def rebatch(batch, device, in_pad_idx, out_pad_idx):
    return Batch(batch.src, batch.trg, device, in_pad_idx, out_pad_idx)