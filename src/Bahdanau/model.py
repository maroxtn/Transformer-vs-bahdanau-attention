import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F



class Encoder(nn.Module): 
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x, inp_length=None):
        
        embedding = self.dropout(self.embedding(x))
        
        if inp_length == None:
            encoder_states, hidden = self.rnn(embedding)
        else:      
            packed = pack_padded_sequence(embedding, inp_length.cpu()) #To speed up training
            encoder_states, hidden = self.rnn(packed)
            encoder_states, _ = pad_packed_sequence(encoder_states)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))

        return encoder_states, hidden



class Decoder(nn.Module):
    
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        
        self.fc_key = nn.Linear(hidden_size, hidden_size)
        self.fc_query = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x, encoder_states, hidden, source, inp_mask):
        
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))

        
        key = self.fc_key(hidden)
        query = self.fc_query(encoder_states)
        
        energy = key+query
        energy = self.energy(torch.tanh(energy))
        
        if inp_mask != None:
            energy = energy.squeeze(-1).masked_fill_(inp_mask, -float('inf')).unsqueeze(-1)

        attention = F.softmax(energy, dim=0) #(seq_len, batch, 1)
                                             #(seq_len, batch, hidden*2)
        
        context_vector = torch.bmm(attention.permute(1, 2, 0), encoder_states.permute(1, 0, 2)).permute(1,0,2)

        #Concatenate the context vector with the embedding of the previous word, and feed it to the GRU
        rnn_input = torch.cat((context_vector, embedding), dim=2)
        outputs, hidden = self.rnn(rnn_input, hidden)

        predictions = self.fc(outputs).squeeze(0)

        return predictions, hidden

        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, inp_length, inp_mask):
        
        batch_size = source.shape[1]
        target_len = target.shape[0]
        
        outputs = []
        
        encoder_states, hidden = self.encoder(source, inp_length)
        
        x = target[0] #<SOS>
        
        for t in range(1, target_len):

            output, hidden = self.decoder(x, encoder_states, hidden, source, inp_mask)

            outputs.append(output)
            best_guess = output.argmax(1)

            x = target[t] #No teacher forcing
            
        
        outputs = torch.cat(outputs)

        return outputs