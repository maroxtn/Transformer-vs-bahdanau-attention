import torch
import time


def run_validation(model, criterion, iterator, device, log_every=100):
    
    model.eval()
    
    total_loss = 0
    
    with torch.no_grad():
    
        for batch_idx, batch in enumerate(iterator):

            inp_data = batch.src.to(device)
            target = batch.trg.to(device)

            output = model(inp_data, target[:-1, ])
            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            loss = criterion(output, target)
            total_loss += loss


    return total_loss/batch_idx



def run_epoch(model, optimizer, criterion, iterator, device, log_every=100):
    
    model.train()
    total_loss = 0
    
    start = time.time()
    n_tokens = 0
    
    print("")
    
    for batch_idx, batch in enumerate(iterator):
        
        inp_data = batch.src
        target = batch.trg

        output = model(inp_data, target)
        output = output.reshape(-1, output.shape[2])

        optimizer.optimizer.zero_grad()
        loss = criterion(output, batch.trg_y)
        total_loss += loss
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        

        n_tokens += batch.ntokens
        if (batch_idx % log_every == 0) and (batch_idx > 0):
            tokens_per_sec = n_tokens/(time.time() - start)
            print(" Step %d - Loss %f - Tokens per Sec %f" % (batch_idx, loss.item(), tokens_per_sec))
            
        
    return total_loss/batch_idx



