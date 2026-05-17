# cross_entropy is a loss function
# formula = -log(p(y|x)) where p(y|x) is the probability of y given x.
# we use cross entropy loss because we are doing classification.

# perplexity as the effective number of choices the model has for the next word.
# eg, Perplexity of 10: The model is as confused as if it were randomly choosing between 10 different words at each step.
#A perfect model that always predicts the next word with 100% confidence would have a perplexity of 1.
# formula = e^cross_entropy 


import torch
import torch.nn as nn
from config import *

def cal_loss_batch(input_batch, target_batch, model, device, z_loss_coeff=Z_LOSS_COEFF):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits, _ = model(input_batch)  # model now returns (logits, kv_cache); discard cache during training
    loss = nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    
    # Z-loss auxiliary term: penalises the log-partition function log(sum(exp(logits))).
    # Without this, softmax logits can grow unboundedly during training because
    # cross-entropy only cares about *differences* between logits, not their
    # absolute magnitude. This causes numerical instability and overconfident
    # predictions. Z-loss keeps logit magnitudes bounded.
    # PaLM uses coefficient 1e-4 (Chowdhery et al., 2022).
    if z_loss_coeff > 0.0:
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = z_loss_coeff * (log_z ** 2).mean()
        loss = loss + z_loss
    
    return loss

def cal_loss_loader(data_loader, model, device, num_batches=None, z_loss_coeff=Z_LOSS_COEFF):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # if num_batches is greater than the number of batches in the dataloader, 
        # set it to the number of batches in the dataloader
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += cal_loss_batch(input_batch, target_batch, model, device,
                                      z_loss_coeff=z_loss_coeff).item()
    return total_loss / num_batches
