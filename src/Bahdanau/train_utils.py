import time
import torch

def run_epoch(model, optimizer, criterion, iterator, log_every=100):
    
    model.train()
    total_loss = 0
    
    start = time.time()
    n_tokens = 0
    
    print("")
    
    for batch_idx, batch in enumerate(iterator):
        
        inp_data = batch.src
        inp_length = batch.src_lengths
        inp_mask = batch.src_mask
        
        target = batch.trg

        output = model(inp_data, target, inp_length, inp_mask)
        
        
        optimizer.zero_grad()
        
        loss = criterion(output, batch.trg_y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        total_loss += loss.item()
        n_tokens += batch.ntokens
        
        
        
        if (batch_idx % log_every == 0) and (batch_idx > 0):
            tokens_per_sec = n_tokens/(time.time() - start)
            print(" Step %d - Loss %f - Tokens per Sec %f" % (batch_idx, loss.item(), tokens_per_sec))
        
    return total_loss / batch_idx



def run_validation(model, optimizer, criterion, iterator):
    
    model.eval()
    total_loss = 0
    
    for batch_idx, batch in enumerate(iterator):
        
        inp_data = batch.src
        inp_length = batch.src_lengths
        inp_mask = batch.src_mask
        
        target = batch.trg

        output = model(inp_data, target, inp_length, inp_mask)
        
        loss = criterion(output, batch.trg_y)
        total_loss += loss.item()
        
    return total_loss / batch_idx