import torch
import numpy as np
import torch.nn.functional as F

def translate_sentence_transformer(model, device, sentence, in_lang_tokenizer, in_lang, out_lang, max_length=50):
    model.eval()
    tokens = [token.text.lower() for token in in_lang_tokenizer(sentence)]

    text_to_indices = [in_lang.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    preds = [out_lang.vocab.stoi[out_lang.init_token]]

    with torch.no_grad():
        
        emb_src = model.encoder(sentence_tensor)
        emb_src = model.pos_encoder(emb_src)

        memory = model.transformer_encoder(emb_src)

        for i in range(50):

            trg = torch.Tensor(preds).long().unsqueeze(1).to(device)
            trg = model.decoder(trg)
            trg = model.pos_decoder(trg)

            out = model.transformer_decoder(tgt = trg, memory = memory)
            out = model.fc_out(out)
            
            

            new = out.squeeze(1)[-1].argmax().item()
            preds.append(new)
            if new == out_lang.vocab.stoi["<eos>"]:
                break

    
    return " ".join([out_lang.vocab.itos[i] for i in preds][1:-1])



def get_out_encoder(src, model, device, in_lang_tokenizer, in_lang):
    
    model.eval()
    tokens = [token.text.lower() for token in in_lang_tokenizer(src)]

    text_to_indices = [in_lang.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)    

    with torch.no_grad():
        
        emb_src = model.encoder(sentence_tensor)
        emb_src = model.pos_encoder(emb_src)

        memory = model.transformer_encoder(emb_src)

        return memory


def beam(phrase, k, model, device, in_lang, out_lang, in_lang_tokenizer, maxlen=50):
    
    model.eval()
    memory = get_out_encoder(phrase, model, device, in_lang_tokenizer, in_lang)

    sos = out_lang.vocab.stoi["<sos>"]
    tgt = [sos]

    with torch.no_grad():

        trg = torch.Tensor(tgt).long().unsqueeze(1).to(device)
        trg = model.decoder(trg)
        trg = model.pos_decoder(trg)

        out = model.transformer_decoder(tgt = trg, memory = memory)
        out = F.softmax(model.fc_out(out), dim=-1)[-1].squeeze()

        args = out.argsort()[-k:].detach().cpu().numpy()
        probs = out[args].detach().cpu().numpy()

        probs = np.log(probs)
        possible = list(zip([tgt + [args[i]] for i in range(k)], probs))
        
        for i in range(maxlen):

            test=  []
            for j in range(k):

                tmp_tgt, tmp_prob = possible[j]

                if tmp_tgt[-1] == out_lang.vocab.stoi["<eos>"]:
                    test.append(possible[j])

                else:
                    trg = torch.Tensor(tmp_tgt).long().unsqueeze(1).to(device)
                    trg = model.decoder(trg)
                    trg = model.pos_decoder(trg)

                    out = model.transformer_decoder(tgt = trg, memory = memory)
                    out = F.softmax(model.fc_out(out), dim=-1)[-1].squeeze()

                    tmp_args = out.argsort()[-k:].detach().cpu().numpy()
                    tmp_probs = out[tmp_args].detach().cpu().numpy()
                    tmp_probs = (tmp_prob + np.log(tmp_probs))/(len(tmp_tgt)-1)

                    for r in range(k): 
                        test.append((tmp_tgt + [tmp_args[r]], tmp_probs[r]))


            possible = sorted(test, key=lambda x:x[1], reverse=True)[:k]
            
    return possible


def convert(x, out_lang):
    
    sentence = x[0]
    sentence = [out_lang.vocab.itos[i] for i in sentence]
    
    return (" ".join(sentence), x[1])