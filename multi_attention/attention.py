# Self-attention processes the entire sequence and produces a new, 
# context-aware embedding for each token that incorporates information from the other relevant tokens in the sequence.

# Query :  What a word wants to know
# Key : What a word offers
# Value : The actual information

# self attention answers the question:-
# For this query (word), which keys (other words) are most relevant, 
# and what values (information) should I gather from them?

# Scores=Q⋅K^T (To find out how relevant each token is to a specific token)
# scaled_score = Scores/√d_k (To normalize the scores), where d_k is the dimension of the key vectors.
# attention_weights = softmax(scaled_score) (To convert the scores into probabilities)
## attention_weights tell us how much each token should contribute to the final output for a specific token.
# output = attention_weights ⋅ V (To compute the weighted sum of the values)


from torch import nn
import torch

class multihead_attention(nn.Module):
    def __init__(self, d_input, d_output, context_length, num_heads, dropout=0.2, qkv_bias=False):
        """
        d_input: embedding dimension of the input tokens
        d_output: embedding dimension of the output tokens
        context_length: length of the sequence
        num_heads: number of heads
        dropout: dropout rate
        qkv_bias: whether to use bias for Q, K, V
        """
        super().__init__()
        # This check ensures that the output dimension can be split evenly among the heads.
        # If d_output is 768 and you have 12 heads, each head will handle a 64-dim vector.
        assert d_output % num_heads == 0, "d_output must be divisible by num_heads"

        self.d_output = d_output
        self.num_heads = num_heads
        self.head_dim = d_output // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_input, d_output, bias=qkv_bias)
        self.W_key = nn.Linear(d_input, d_output, bias=qkv_bias)
        self.W_value = nn.Linear(d_input, d_output, bias=qkv_bias)

        self.out_proj = nn.Linear(d_output, d_output)  # Linear layer to combine head outputs
        
        self.dropout = nn.Dropout(dropout)

        # It creates a permanent "buffer" in the model for the causal attention mask.
        self.register_buffer(   # move to device (cuda if available)
            "mask",
            # Creates a matrix of ones with the shape (context_length, context_length)
            # torch.triu(..., diagonal=1) sets all elements on and below the main diagonal to 0,
            # leaving only the upper triangle as 1s
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
    
    def forward(self, x, past_keys=None, past_values=None):
        """
        x: input tensor (batch, num_tokens, d_in)
        past_keys: cached keys from previous steps (batch, num_heads, past_len, head_dim) or None
        past_values: cached values from previous steps (batch, num_heads, past_len, head_dim) or None

        During training: past_keys/past_values are None, full sequence processed with causal mask.
        During inference with KV cache: past_keys/past_values hold all previous tokens' K/V.
           Only the single new token is in x. New K/V are computed and concatenated with cache.
           This avoids O(n²) recomputation — each step only does O(n) attention.
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (batch_size, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim) for parallel processing
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # KV cache: append new K/V to cached K/V
        if past_keys is not None and past_values is not None:
            # Concatenate along the sequence-length dimension (dim=2)
            # past_keys: (b, num_heads, past_len, head_dim)
            # keys:      (b, num_heads, 1, head_dim)  — only the new token
            # result:    (b, num_heads, past_len+1, head_dim)
            keys = torch.cat([past_keys, keys], dim=2)
            values = torch.cat([past_values, values], dim=2)

        # Full sequence length (past_len + new_tokens) — used for the causal mask
        full_seq_len = keys.size(2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Causal masking 
        # During training (past_keys=None, num_tokens>1): mask future tokens.
        # During inference with KV cache (past_keys != None, num_tokens=1): no future tokens to mask.
        if past_keys is None and num_tokens > 1:
            # Training / first-pass inference: mask so token i can't see token j>i
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # top left upper triangle
            attn_scores.masked_fill_(mask_bool, -torch.inf)
        # (If past_keys is not None, we are generating 1 token at a time,
        #  so there are no future tokens to mask — the new token attends to all past + itself.)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)   # softmax(attn_scores / sqrt(head_dim))
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_output)
        context_vec = self.out_proj(context_vec) # optional projection

        # The output is a new (2, 512, 768) tensor where each token's vector is now context-aware, 
        # containing information from itself and all previous tokens.

        # Return updated keys/values so the caller can cache them for the next generation step
        return context_vec, keys, values
