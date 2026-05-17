import torch
import torch.nn as nn
from config import *
from model.architecture import TransformerBlock, RMSNorm

class kaitomodel(nn.Module):
    def __init__(self):
        super().__init__()
        # Token embedding: maps vocabulary indices to dense vectors.
        # Positional information is now injected inside attention via RoPE,
        # so we don't need a learned position embedding table anymore.
        # This saves 512 * 768 = 393K parameters and removes the hard
        # max-length constraint (RoPE extrapolates to arbitrary lengths).
        self.token_embedding = nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
        # Reinitialise embedding weights with smaller std (0.02 instead of N(0,1)).
        # With weight tying (below), the embedding weight is also used as the output
        # projection. Default Embedding init (N(0,1)) produces logit std ≈ sqrt(768) ≈ 27.7,
        # making initial cross-entropy loss ~700 instead of ~log(50257) ≈ 10.8.
        # Std 0.02 matches GPT-2/Llama convention and gives reasonable logit scales.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        self.dropout = nn.Dropout(DROPOUT)
        # ModuleList (not Sequential) so we can pass KV-cache through each block individually
        # Each transformer block can optionally use MoE (sparse FFN) instead
        # of a dense FFN. The USE_MOE flag controls this globally.
        self.trf_block = nn.ModuleList(
            [TransformerBlock(use_moe=USE_MOE) for _ in range(N_LAYERS)]
        )
        # RMSNorm is faster than LayerNorm (~15-25% fewer FLOPs) and empirically
        # equivalent — used by Llama, Mistral, Gemma.
        self.final_norm = RMSNorm(OUTPUT_DIM)
        self.out_head = nn.Linear(OUTPUT_DIM, VOCAB_SIZE, bias=False)
        
        # Weight tying: share the weight matrix between token_embedding and out_head.
        # The embedding layer learns a "meaning space" for tokens, and the output
        # head projects to the same space. Tying ensures consistency: a token's
        # embedding is the representation that should cause the model to predict
        # that token. This also saves ~38.6M parameters (the entire output head).
        self.out_head.weight = self.token_embedding.weight
    
    def forward(self, input_ids, past_key_values=None):
        """
        input_ids: (batch_size, seq_len)
        past_key_values: list of (keys, values) tuples, one per layer, or None.
                         Each keys tensor has shape (batch, n_kv_heads, past_len, head_dim).
                         Each values tensor has shape (batch, n_kv_heads, past_len, head_dim).

        Returns: (logits, new_key_values)
                 logits: (batch_size, seq_len, vocab_size)
                 new_key_values: list of (keys, values) for each layer to cache.

        During training: past_key_values is None, full sequence processed.
        During inference with KV cache: past_key_values holds all previous layers' caches.
        """
        batch_size, seq_len = input_ids.shape

        # Safety guard: truncate excessively long sequences.
        # With RoPE this is no longer strictly required (RoPE extrapolates),
        # but it prevents silent OOM from absurdly long inputs.
        if seq_len > MAX_LENGTH:
            input_ids = input_ids[:, -MAX_LENGTH:]
            seq_len = MAX_LENGTH

        # Token embeddings only — RoPE inside the attention layers provides
        # positional information, so we don't add learned position embeddings.
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, embed_dim]

        x = self.dropout(x)

        # KV-cache: process each block with cached keys/values from prior steps
        new_key_values = []
        for i, block in enumerate(self.trf_block):
            if past_key_values is not None:
                # Retrieve cached K/V for this layer
                pk, pv = past_key_values[i]
            else:
                pk = pv = None
            x, k, v = block(x, pk, pv)
            new_key_values.append((k, v))

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits, new_key_values

    def get_num_layers(self):
        """Number of transformer blocks — used by generate_text to initialise KV cache."""
        return len(self.trf_block)
