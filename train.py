import torch
from config import *
from main import kaitomodel
from loss.cal_loss import cal_loss_loader, cal_loss_batch
from text_generation import generate_text

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = cal_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = cal_loss_loader(val_loader, model, device, num_batches = eval_iter)
    model.train()
    return train_loss, val_loss

def train_model(model,train_loader, val_loader, optimizer, device, num_epochs, 
                eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0,-1
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # reset loss gradient form the previous batch iteration
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen+=input_batch.numel()
            global_step+=1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                print(f"Epoch {epoch}, Step {global_step}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        
        generated_text = generate_text(model, start_context, tokenizer, new_max_length=10)
        print(f"Generated Text: {generated_text}")
    return train_losses, val_losses, track_tokens_seen