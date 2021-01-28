import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F

import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)



class TransformerModel(nn.Module):
    
    def __init__(self, intoken, outtoken ,hidden, in_pad_idx, out_pad_idx, enc_layers=2, dec_layers=2, dropout=.1, nheads=2, ff_model=128):
        super(TransformerModel, self).__init__()
        
        self.encoder = nn.Embedding(intoken, hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Embedding(outtoken, hidden) 
        self.pos_decoder = PositionalEncoding(hidden, dropout)
        
        
        encoder_layers = TransformerEncoderLayer(d_model=hidden, nhead = nheads, dim_feedforward = ff_model, dropout=dropout, activation='relu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, enc_layers)

        encoder_layers = TransformerDecoderLayer(hidden, nheads, ff_model, dropout, activation='relu')
        self.transformer_decoder = TransformerDecoder(encoder_layers, dec_layers)        

        self.fc_out = nn.Linear(hidden, outtoken)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None


        self.in_pad_idx = in_pad_idx
        self.out_pad_idx = out_pad_idx
        
    def generate_square_subsequent_mask(self, sz, sz1=None):
        
        if sz1 == None:
            mask = torch.triu(torch.ones(sz, sz), 1)
        else:
            mask = torch.triu(torch.ones(sz, sz1), 1)
            
        return mask.masked_fill(mask==1, float('-inf'))

    def make_len_mask_enc(self, inp):
        return (inp == self.in_pad_idx).transpose(0, 1)   #(batch_size, output_seq_len)
    
    def make_len_mask_dec(self, inp):
        return (inp == self.out_pad_idx).transpose(0, 1) #(batch_size, input_seq_len)
    


    def forward(self, src, trg): #SRC: (seq_len, batch_size)

        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)
            

        #Adding padding mask
        src_pad_mask = self.make_len_mask_enc(src)
        trg_pad_mask = self.make_len_mask_dec(trg)
             

        #Add embeddings Encoder
        src = self.encoder(src)  #Embedding, (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)   #Pos embedding
        
        
        #Add embedding decoder
        trg = self.decoder(trg) #(seq_len, batch_size, d_model)
        trg = self.pos_decoder(trg)

        
        memory = self.transformer_encoder(src, None, src_pad_mask)
        output = self.transformer_decoder(tgt = trg, memory = memory, tgt_mask = self.trg_mask, memory_mask = None, 
                                          tgt_key_padding_mask = trg_pad_mask, memory_key_padding_mask = src_pad_mask)

        output = self.fc_out(output)

        return output



def create_model(input_size_encoder, input_size_decoder ,d_model, in_pad_idx, out_pad_idx, enc_layers, dec_layers, dropout, nheads, ff_model):
    return TransformerModel(input_size_encoder, input_size_decoder ,d_model, in_pad_idx, out_pad_idx, enc_layers, dec_layers, dropout, nheads, ff_model)