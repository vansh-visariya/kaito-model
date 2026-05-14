import torch
from config import *
from main import kaitomodel
from loss.cal_loss import cal_loss_loader, cal_loss_batch
from text_generation import generate_text

def evaluate_model(model, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        val_loss = cal_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return val_loss

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1
    running_train_loss = 0.0
    running_train_count = 0
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            token_seen += input_batch.numel()
            global_step += 1
            running_train_loss += loss.item()
            running_train_count += 1

            if global_step % eval_freq == 0:
                avg_train_loss = running_train_loss / running_train_count
                val_loss = evaluate_model(model, val_loader, device, eval_iter)
                train_losses.append(avg_train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                running_train_loss = 0.0
                running_train_count = 0
                print(f"Epoch {epoch}, Step {global_step}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        generated_text = generate_text(model, start_context, tokenizer, new_max_length=10)
        print(f"Generated Text: {generated_text}")

    torch.save(model.state_dict(), "model.pt")
    print("Model saved to model.pt")
    return train_losses, val_losses, track_tokens_seen