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


def _create_adamw_optimizer(model):
    """
    Create an AdamW optimizer with decoupled weight decay.
    
    Weight decay is applied only to weight matrices, not to biases or norm
    parameters. Applying decay to biases pushes them toward zero and conflicts
    with their role as offsets. Norm parameters (scale) interact multiplicatively
    — shrinking them forces the rest of the model to compensate, which hurts
    convergence. This "no decay for biases/norms" convention is standard
    (GPT-3, Llama, PaLM all use it).
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Skip weight decay for: biases, norm scales, and the tied embedding
        # weight (which is shared with out_head and should be treated as an
        # embedding — embedding regularisation is handled by the softmax itself).
        if 'bias' in name or 'norm' in name or 'scale' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),   # Llama-style: 0.9 momentum, 0.95 RMS (vs default 0.999)
        eps=1e-8
    )


def _create_warmup_cosine_scheduler(optimizer, total_opt_steps, warmup_steps):
    """
    Linear warmup + cosine decay scheduler.
    
    LR starts at 0, linearly increases to LEARNING_RATE over warmup_steps,
    then follows a cosine curve down to 0 over the remaining steps.
    
    Why warmup?
        Adam's momentum/variance estimates (m, v) are initialised at 0 and need
        time to stabilise. Starting at full LR before these estimates converge
        can cause gradient explosions in deep transformers (Priya Goyal et al., 2017).
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: progress from 0 to 1
            return float(current_step + 1) / float(max(1, warmup_steps))
        else:
            # Cosine decay from 1 to 0
            progress = float(current_step - warmup_steps) / float(
                max(1, total_opt_steps - warmup_steps)
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(model, train_loader, val_loader, device, num_epochs,
                eval_freq, eval_iter, start_context, tokenizer):
    """
    Full training loop with:
      - AdamW optimizer with decoupled weight decay
      - Linear LR warmup + cosine annealing (per-step, not per-epoch)
      - Gradient accumulation (effective batch = BATCH_SIZE * ACCUMULATION_STEPS)
      - Gradient clipping (prevents gradient explosion in deep transformers)
      - Z-loss auxiliary term for logit stabilisation
      - Periodic evaluation + text generation
      - Model checkpointing at the end
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    token_seen, global_step = 0, -1
    running_train_loss = 0.0
    running_train_count = 0

    # AdamW with proper parameter grouping — built internally so callers
    # always get the correct configuration (weight decay, betas).
    optimizer = _create_adamw_optimizer(model)

    # Precompute total and warmup steps for the LR scheduler.
    # Total optimizer steps = total batches / ACCUMULATION_STEPS
    # (each optimizer step covers ACCUMULATION_STEPS batches).
    total_batches = len(train_loader) * num_epochs
    total_opt_steps = total_batches // ACCUMULATION_STEPS + 1
    warmup_steps = int(WARMUP_RATIO * total_opt_steps)

    # Linear warmup + cosine decay, stepped per optimizer step.
    scheduler = _create_warmup_cosine_scheduler(optimizer, total_opt_steps, warmup_steps)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # zero once at the start of each epoch

        for i, (input_batch, target_batch) in enumerate(train_loader):
            # Forward & loss (includes Z-loss auxiliary term, configured in config.py)
            loss = cal_loss_batch(input_batch, target_batch, model, device,
                                  z_loss_coeff=Z_LOSS_COEFF)

            # Collect MoE load-balancing auxiliary losses from all blocks.
            # Each MoE layer stores its aux loss after the forward pass;
            # we sum them here and add to the total loss.
            if USE_MOE:
                moe_aux_loss = 0.0
                for block in model.trf_block:
                    if block.use_moe:
                        moe_aux_loss = moe_aux_loss + block.feedforward.last_aux_loss
                loss = loss + MOE_LOSS_COEFF * moe_aux_loss

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
                scheduler.step()  # step LR after each optimizer step

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

        # End-of-epoch: generate sample text
        generated_text = generate_text(
            model, start_context, tokenizer,
            new_max_length=10,
            temperature=0.8,   # slight randomness for more interesting samples
            top_k=TOP_K,
            top_p=TOP_P
        )
        print(f"Generated Text: {generated_text}")

    # Save model 
    torch.save(model.state_dict(), "model.pt")
    print("Model saved to model.pt")
    return train_losses, val_losses, track_tokens_seen