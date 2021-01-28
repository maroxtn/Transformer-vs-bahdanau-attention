import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IWSLT
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
import time
import math 

import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

import spacy.cli
import en_core_web_sm
import de_core_news_sm


from Bahdanau.model import Seq2Seq, Encoder, Decoder
from Bahdanau.train_utils import run_epoch, run_validation
from Bahdanau.utils import translate_sentence_bahdanau
from Bahdanau.batch import rebatch

import yaml



with open('config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


CFG = {"IN_LANG":"de", "OUT_LANG": "en"}

spacy.cli.download("en_core_web_sm")
spacy.cli.download("de_core_news_sm")


if CFG["IN_LANG"] == "en":
    spacy_in_lang = en_core_web_sm.load()
    spacy_out_lang = de_core_news_sm.load()
else:
    spacy_in_lang = de_core_news_sm.load()
    spacy_out_lang = en_core_web_sm.load()
    

def tokenizer_in(text):
    return [tok.text for tok in spacy_in_lang.tokenizer(text)]

def tokenizer_out(text):
    return [tok.text for tok in spacy_out_lang.tokenizer(text)]

in_lang = Field(tokenize=tokenizer_in, lower=True, include_lengths=True)
out_lang = Field(tokenize=tokenizer_out, lower=True, init_token="<sos>", eos_token="<eos>", include_lengths=True)




MAX_LEN = config["training_phrases_max_len"]



train_data, valid_data, test_data = IWSLT.splits(root="data",
        exts=("."+CFG["IN_LANG"], "."+CFG["OUT_LANG"]), fields=(in_lang, out_lang ),filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
            len(vars(x)['trg']) <= MAX_LEN)


in_lang.build_vocab(train_data, min_freq=config["min_freq_words"])
out_lang.build_vocab(train_data, min_freq=config["min_freq_words"])



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Training Hyperparameters
num_epochs = config["epochs"]
lr = config["rnn_lr"]
batch_size = config["rnn_batch_size"]
d_model = config["d_model"]

input_size_encoder = len(in_lang.vocab)
input_size_decoder = len(out_lang.vocab)
output_size = len(out_lang.vocab)


encoder_embedding_size = d_model
decoder_embedding_size = d_model
hidden_size = d_model

num_layers = 1  #Do not change this, it would break the model
dropout = 0.1


in_pad_idx = in_lang.vocab.stoi['<pad>']
out_pad_idx = out_lang.vocab.stoi['<pad>']

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)



encoder_net = Encoder(input_size_encoder, encoder_embedding_size, 
                      hidden_size, num_layers, dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size, 
                      hidden_size, output_size, num_layers, dropout).to(device)


model = Seq2Seq(encoder_net, decoder_net).to(device)


optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=out_pad_idx)



best_loss = 65646

for epoch in range(num_epochs):
    
    print(f'Epoch [{epoch} / {num_epochs}]')


    loss =  run_epoch(model, optimizer, criterion, (rebatch(b, device, in_pad_idx, out_pad_idx) for b in train_iterator))
    validation_loss = run_validation(model, optimizer, criterion, (rebatch(b, device, in_pad_idx, out_pad_idx) for b in valid_iterator))
    
    
    rand_i01 = np.random.randint(0, len(train_data))
    rand_i02 = np.random.randint(0, len(valid_data))
    rand_i03 = np.random.randint(0, len(test_data))
    
    sentence01, expected01 = " ".join(train_data[rand_i01].src), " ".join(train_data[rand_i01].trg)
    sentence02, expected02 = " ".join(valid_data[rand_i02].src), " ".join(valid_data[rand_i02].trg)
    sentence03, expected03 = " ".join(test_data[rand_i03].src), " ".join(test_data[rand_i03].trg)
    
    translated_sentence01 = translate_sentence_bahdanau(model, sentence01, spacy_in_lang, in_lang, out_lang, device, max_length=50)
    translated_sentence02 = translate_sentence_bahdanau(model, sentence02, spacy_in_lang, in_lang, out_lang, device, max_length=50)
    translated_sentence03 = translate_sentence_bahdanau(model, sentence03, spacy_in_lang, in_lang, out_lang, device, max_length=50)
    #out = beam(sentence, 3)  list(map(convert, out[:2]))
    
    
    print(f"\nExample #1 (from Train data): \nTranslation: { translated_sentence01 }\nExpected: { expected01 }")
    print(f"\nExample #2 (from Validation): \nTranslation: { translated_sentence02 }\nExpected: { expected02 }")
    print(f"\nExample #3 (from Test data): \nTranslation: { translated_sentence03 }\nExpected: { expected03 }\n")
    
    print(f"\n Train loss {loss} | Validation loss {validation_loss} \n\n\n")
    
    if validation_loss < best_loss:
        torch.save(model.state_dict(), "../models/rnn_model_new")
        best_loss = validation_loss