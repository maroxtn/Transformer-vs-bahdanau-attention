import torch

def translate_sentence_bahdanau(model, sentence, in_lang_tokenizer, in_lang, out_lang, device, max_length=50):
    
    model.eval()

    tokens = [token.text.lower() for token in in_lang_tokenizer(sentence)]

    text_to_indices = [in_lang.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    preds = [out_lang.vocab.stoi[out_lang.init_token]]

    with torch.no_grad():
        
        encoder_states, hidden = model.encoder(sentence_tensor)
        
        for t in range(max_length):
                    
            trg = torch.Tensor([preds[-1]]).long().to(device)

            output, hidden = model.decoder(trg, encoder_states, hidden, sentence_tensor, None)
            new = output.argmax(1).item()
            
            preds.append(new)
            
            if new == out_lang.vocab.stoi["<eos>"]:
                break
            
        
    return " ".join([out_lang.vocab.itos[i] for i in preds][1:-1])


def beam(phrase, model, out_lang, in_lang, in_lang_tokenizer, k):  #K: beam width
    
    model.eval()
    
    sos = out_lang.vocab.stoi["<sos>"]
    tgt = [sos]
    
    #Prepare sentence
    tokens = [token.text.lower() for token in in_lang_tokenizer(phrase)]
    tokens.append(in_lang.eos_token)
    tokens.insert(0, in_lang.init_token)

    text_to_indices = [in_lang.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)    
    

    with torch.no_grad():

        #Get encoder output
        encoder_states, hidden = model.encoder(sentence_tensor)
        
        
        #Get first output from model
        trg = torch.Tensor([tgt[-1]]).long().to(device)

        output, hidden = model.decoder(trg, encoder_states, hidden,sentence_tensor)
        out = F.softmax(output).squeeze()



        args = out.argsort()[-k:]
        probs = out[args].detach().cpu().numpy()
        
        args = args.detach().cpu().numpy()
        
        
        probs = np.log(probs)
        possible = list(zip([tgt + [args[i]] for i in range(k)], probs, [hidden.clone() for j in range(k)]))


        for i in range(50):

            test=  []
            for j in range(k):

                tmp_tgt, tmp_prob, tmp_hidden = possible[j]

                if tmp_tgt[-1] == out_lang.vocab.stoi["<eos>"]:  #If sentence already ended
                    test.append(possible[j])

                else:
                    
                    #Compute output
                    trg = torch.Tensor([tmp_tgt[-1]]).long().to(device)

                    output, hidden = model.decoder(trg, encoder_states, tmp_hidden, sentence_tensor)
                    out = F.softmax(output).squeeze()
                    
                    
                    tmp_args = out.argsort()[-k:]
                    tmp_probs = out[args].detach().cpu().numpy()

                    tmp_args = tmp_args.detach().cpu().numpy()
                    tmp_probs = (tmp_prob + np.log(tmp_probs))/(len(tmp_tgt)-1)


                    for r in range(k): 
                        test.append((tmp_tgt + [tmp_args[r]], tmp_probs[r], hidden))


            possible = sorted(test, key=lambda x:x[1], reverse=True)[:k]


                    
    
    return possible



def convert(x, out_lang):
    
    sentence = x[0]
    sentence = [out_lang.vocab.itos[i] for i in sentence]
    
    return (" ".join(sentence), x[1])