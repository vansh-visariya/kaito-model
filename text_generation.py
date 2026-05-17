from config import *
import torch

def _apply_top_k_filtering(logits, top_k):
    """
    Keep only the top_k tokens by logit value; set the rest to -inf so softmax zeroes them out.
    Reduces the sampling pool to the k most likely tokens (Fan et al., 2018; arXiv:1805.04833).
    This prevents the model from choosing extremely improbable tokens.
    """
    top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
    # Mask out all tokens below the k-th highest logit
    logits = torch.where(logits < top_k_vals[:, -1:], float('-inf'), logits)
    return logits


def _apply_top_p_filtering(logits, top_p):
    """
    Nucleus (top-p) sampling: keep the smallest set of tokens whose cumulative probability
    exceeds top_p. Dynamically adapts the pool size per step (Holtzman et al., 2019; arXiv:1904.09751).
    Tokens outside the nucleus are masked to -inf.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the mask right by one so the first token that crosses the threshold is kept
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    # Scatter back to the original index order
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits


def generate_text(model, prompt, tokenizer,
                  new_max_length=200,
                  temperature=TEMPERATURE,
                  top_k=TOP_K,
                  top_p=TOP_P):
    """
    Autoregressive text generation with KV-cache (O(n) per step instead of O(n²))
    and configurable sampling strategy.

    Sampling priority (applied in order):
      1. Scale logits by temperature (0 = argmax / greedy).
      2. If top_k > 0, keep only the top-k tokens.
      3. If top_p < 1.0, keep the nucleus (top-p) tokens.
      4. Sample from the filtered distribution via torch.multinomial.

    KV-cache mechanism:
      - Step 0: process the full prompt, caching every layer's K/V.
      - Step 1+: only feed the single *new* token; the attention layer concatenates
        its K/V to the cache, avoiding O(n²) recomputation of the full prefix.
    """
    model.eval()

    # Tokenize the prompt once and keep raw IDs (no decode/re-encode drift)
    token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

    past_key_values = None  # Will become a list of (keys, values) tuples, one per layer

    for step in range(new_max_length):
        with torch.no_grad():
            # Guard against sequence exceeding the learned position embeddings
            if token_ids.size(1) > MAX_LENGTH:
                token_ids = token_ids[:, -MAX_LENGTH:]
                # Truncation invalidates the KV cache — reset it
                past_key_values = None

            if step == 0:
                # First step: process full prompt, initialise KV cache
                logits, past_key_values = model(token_ids, past_key_values)
            else:
                # Subsequent steps: only pass the single *new* token.
                # The attention layers concatenate its K/V with the cached prefix.
                logits, past_key_values = model(token_ids[:, -1:], past_key_values)

            # Focus on the last token's logits (the one we're predicting)
            logits = logits[:, -1, :]

            # ---- Sampling strategy ----
            # 1. Temperature scaling (lower = sharper, 0 = greedy)
            if temperature == 0.0:
                # argmax (greedy) — no randomness
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # 2. Top-k filtering
                if top_k is not None and top_k > 0:
                    logits = _apply_top_k_filtering(logits, top_k)

                # 3. Top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    logits = _apply_top_p_filtering(logits, top_p)

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append the predicted token to the sequence
            token_ids = torch.cat([token_ids, next_token], dim=1)

    model.train()
    return tokenizer.decode(token_ids.squeeze(0).tolist())
