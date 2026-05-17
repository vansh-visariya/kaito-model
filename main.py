import torch
import torch.nn as nn
from config import *
from model.architecture import TransformerBlock, LayerNorm

class kaitomodel(nn.Module):
    def __init__(self):
        super().__init__()
        # Add embedding layers to the model
        self.token_embedding = nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
        self.pos_embedding = nn.Embedding(MAX_LENGTH, OUTPUT_DIM)
        
        self.dropout = nn.Dropout(DROPOUT)
        # ModuleList (not Sequential) so we can pass KV-cache through each block individually
        self.trf_block = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.final_norm = LayerNorm(OUTPUT_DIM)
        self.out_head = nn.Linear(OUTPUT_DIM, VOCAB_SIZE, bias=False)
    
    def forward(self, input_ids, past_key_values=None):
        """
        input_ids: (batch_size, seq_len)
        past_key_values: list of (keys, values) tuples, one per layer, or None.
                         Each keys/values tensor has shape (batch, n_heads, past_len, head_dim).

        Returns: (logits, new_key_values)
                 logits: (batch_size, seq_len, vocab_size)
                 new_key_values: list of (keys, values) for each layer to cache.

        During training: past_key_values is None, full sequence processed.
        During inference with KV cache: past_key_values holds all previous layers' caches.
        """
        batch_size, seq_len = input_ids.shape

        # Truncate to max supported sequence length to prevent position embedding OOB
        if seq_len > MAX_LENGTH:
            input_ids = input_ids[:, -MAX_LENGTH:]
            seq_len = MAX_LENGTH

        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch_size, seq_len, embed_dim]

        # Get position embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_embeds = self.pos_embedding(positions)  # [seq_len, embed_dim]

        # Combine embeddings
        x = token_embeds + pos_embeds  # Broadcasting: [batch_size, seq_len, embed_dim]

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
