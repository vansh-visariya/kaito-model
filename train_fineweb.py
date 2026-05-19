print("Starting train_fineweb.py")
import datasets
import torch
import tiktoken
import time
from config import *
from main import kaitomodel
from data_prep.fineweb_loader import create_fineweb_dataloaders
from train import _create_adamw_optimizer, _create_warmup_cosine_scheduler
from loss.cal_loss import cal_loss_batch, cal_loss_loader
from text_generation import generate_text

def evaluate_model_streaming(model, val_loader, device, eval_iter):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_iter = iter(val_loader)
        for _ in range(eval_iter):
            try:
                input_batch, target_batch = next(val_iter)
            except StopIteration:
                break
            loss = cal_loss_batch(input_batch, target_batch, model, device, z_loss_coeff=0.0)
            val_loss += loss.item()
    model.train()
    return val_loss / eval_iter if eval_iter > 0 else float('inf')


def train_streaming(max_steps=5000, eval_freq=500, eval_iter=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Initialize the Fineweb streaming dataloaders
    print("Initializing FineWeb streaming dataloaders...")
    train_loader, val_loader = create_fineweb_dataloaders(tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, stride=MAX_LENGTH)
    
    # Initialize the model
    model = kaitomodel()
    model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = _create_adamw_optimizer(model)
    
    # Precompute steps
    total_opt_steps = max_steps // ACCUMULATION_STEPS + 1
    warmup_steps = int(WARMUP_RATIO * total_opt_steps)
    
    scheduler = _create_warmup_cosine_scheduler(optimizer, total_opt_steps, warmup_steps)
    
    # Streaming training loop
    model.train()
    optimizer.zero_grad()
    
    running_train_loss = 0.0
    running_train_count = 0
    token_seen = 0
    start_time = time.time()
    
    print(f"Starting training for {max_steps} steps...")
    
    train_iter = iter(train_loader)
    
    for step in range(1, max_steps + 1):
        try:
            input_batch, target_batch = next(train_iter)
        except StopIteration:
            print("Reached the end of the streaming dataset!")
            break
            
        loss = cal_loss_batch(input_batch, target_batch, model, device, z_loss_coeff=Z_LOSS_COEFF)
        
        if USE_MOE:
            moe_aux_loss = 0.0
            for block in model.trf_block:
                if block.use_moe:
                    moe_aux_loss += block.feedforward.last_aux_loss
            loss += MOE_LOSS_COEFF * moe_aux_loss
            
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        
        token_seen += input_batch.numel()
        running_train_loss += loss.item() * ACCUMULATION_STEPS
        running_train_count += 1
        
        if step % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        if step % eval_freq == 0:
            avg_train_loss = running_train_loss / running_train_count
            val_loss = evaluate_model_streaming(model, val_loader, device, eval_iter)
            
            elapsed = time.time() - start_time
            print(f"Step {step} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Tokens Seen: {token_seen:,} | Time: {elapsed:.1f}s")
            
            # Generate sample text
            start_context = "In the beginning"
            generated_text = generate_text(model, start_context, tokenizer, new_max_length=15, temperature=0.8, top_k=TOP_K, top_p=TOP_P)
            print(f"Generated: {generated_text}")
            
            # Reset running stats
            running_train_loss = 0.0
            running_train_count = 0
            start_time = time.time()
            
            # Save periodic checkpoint
            torch.save(model.state_dict(), "model.pt")

    # Final save
    torch.save(model.state_dict(), "model.pt")
    print("Training finished. Model saved to model.pt")

if __name__ == "__main__":
    print("main start")
    # A full run on 10B tokens at Batch Size=2 and Max Length=512 is ~9.7M steps.
    # Set to a very large number (e.g. 2,000,000) and evaluate every 2000 steps.
    train_streaming(max_steps=2000000, eval_freq=2000, eval_iter=20)
