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
    """
    Full training loop with:
      - Gradient accumulation (effective batch = BATCH_SIZE * ACCUMULATION_STEPS)
      - Gradient clipping (prevents gradient explosion in deep transformers)
      - Cosine annealing LR scheduler (better convergence than constant LR)
      - Periodic evaluation + text generation
      - Model checkpointing at the end
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1
    running_train_loss = 0.0
    running_train_count = 0

    # Cosine annealing scheduler: decays LR from initial to 0 over num_epochs
    # This mimics the common "warmup + decay" schedule used in GPT-2 training,
    # without requiring a warmup phase. The LR drops smoothly, allowing larger
    # updates early and fine-grained convergence later.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # zero once at the start of each epoch

        for i, (input_batch, target_batch) in enumerate(train_loader):
            # Forward & loss
            loss = cal_loss_batch(input_batch, target_batch, model, device)

            # Gradient accumulation
            # Divide by ACCUMULATION_STEPS so that after ACCUMULATION_STEPS
            # backward passes the total gradient matches what a single large
            # batch would produce. This allows effective batch sizes larger
            # than what fits in GPU memory.
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            token_seen += input_batch.numel()
            global_step += 1
            # Track undivided loss for reporting
            running_train_loss += loss.item() * ACCUMULATION_STEPS
            running_train_count += 1

            # Optimiser step (only every ACCUMULATION_STEPS batches)
            if (i + 1) % ACCUMULATION_STEPS == 0:
                # Gradient clipping: cap the L2 norm of all gradients to GRAD_CLIP_MAX_NORM.
                # Without this, transformers are prone to gradient explosion during early
                # training, which can cause NaN loss or divergence.
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                optimizer.step()
                optimizer.zero_grad()

            # Evaluation (periodic)
            if global_step % eval_freq == 0:
                avg_train_loss = running_train_loss / running_train_count
                val_loss = evaluate_model(model, val_loader, device, eval_iter)
                train_losses.append(avg_train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(token_seen)
                running_train_loss = 0.0
                running_train_count = 0
                print(f"Epoch {epoch}, Step {global_step}, "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # End-of-epoch: generate sample text & step scheduler
        generated_text = generate_text(
            model, start_context, tokenizer,
            new_max_length=10,
            temperature=0.8,   # slight randomness for more interesting samples
            top_k=TOP_K,
            top_p=TOP_P
        )
        print(f"Generated Text: {generated_text}")
        scheduler.step()  # decay learning rate

    # Save model 
    torch.save(model.state_dict(), "model.pt")
    print("Model saved to model.pt")
    return train_losses, val_losses, track_tokens_seen