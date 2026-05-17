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


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Rotates Query and Key vectors by a frequency that depends on position,
    so the dot-product Q·K^T naturally encodes relative position (m-n).

    Why replace learned position embeddings?
        Learned embeddings are fixed at MAX_LENGTH — sequences beyond that
        either crash or truncate silently. RoPE operates on *relative*
        position via rotation, so it extrapolates to arbitrary sequence
        lengths without extra parameters. This is also parameter-free
        (no 512*768=393K embedding table to learn).

    Why precompute cos/sin tables?
        Computing cos(m*theta_i) on every forward pass is wasteful.
        Since theta_i is fixed at init, precompute up to max_seq_len
        and slice during forward. This adds 2*max_seq_len*head_dim floats
        but avoids expensive trig ops at runtime.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        # RoPE requires head_dim to be even because we rotate pairs (2i, 2i+1)
        assert head_dim % 2 == 0, "RoPE requires head_dim to be even"

        # Compute the frequency for each dimension pair.
        # theta_i = base^(-2i / head_dim) for i in [0, head_dim/2)
        # This creates a geometric progression. The choice of base=10000 follows
        # "Attention Is All You Need" and ensures a wide range of frequencies:
        # high frequencies capture local patterns, low frequencies capture long-range.
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Precompute position indices: [0, 1, 2, ..., max_seq_len - 1]
        positions = torch.arange(max_seq_len).float()

        # Outer product: (max_seq_len,) x (head_dim/2,) -> (max_seq_len, head_dim/2)
        angles = torch.outer(positions, freqs)

        # Duplicate to fill the full head_dim: (max_seq_len, head_dim)
        # Both (2i) and (2i+1) get the same angle, forming a 2D rotation matrix.
        angles = torch.cat([angles, angles], dim=-1)

        # Precompute cos and sin tables — used in forward without any trig ops
        self.register_buffer("cos", angles.cos())  # (max_seq_len, head_dim)
        self.register_buffer("sin", angles.sin())  # (max_seq_len, head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_kv_heads, seq_len, head_dim)
            position_ids: (seq_len,) — absolute positions for this batch

        Returns:
            q_rotated, k_rotated: same shapes as inputs
        """
        # Gather cos/sin for the requested positions
        cos = self.cos[position_ids].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = self.sin[position_ids].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

        # Apply pair-wise rotation:
        # RoPE(x)_(2i)   = x_(2i)   * cos - x_(2i+1) * sin
        # RoPE(x)_(2i+1) = x_(2i+1) * cos + x_(2i)   * sin
        #
        # This is derived from the 2D rotation matrix applied independently
        # to each pair (2i, 2i+1). The dot-product of two such rotated vectors
        # depends only on (m-n), giving relative-position bias.
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Swaps and negates the first half of the last dimension.
        
        Equivalent to: [-x1, x0, -x3, x2, ...] which implements the
        second term of the 2D rotation. Splitting the dimension in half
        and swapping is more GPU-friendly than indexing pairs.
        """
        x_half = x.shape[-1] // 2
        x1 = x[..., :x_half]   # even-indexed dims
        x2 = x[..., x_half:]   # odd-indexed dims
        return torch.cat((-x2, x1), dim=-1)


class grouped_query_attention(nn.Module):
    """
    Grouped Query Attention (GQA) with Rotary Position Embeddings (RoPE).

    GQA extends multi-head attention by allowing n_kv_heads < n_heads.
    Query heads are partitioned into groups; each group shares one KV head.

    When n_kv_heads == n_heads: standard Multi-Head Attention (MHA).
    When n_kv_heads <  n_heads: Grouped Query Attention (GQA).
    When n_kv_heads == 1:       Multi-Query Attention (MQA).

    Why GQA instead of full MHA?
        The KV cache is the dominant memory consumer during long-sequence
        generation. With 12 KV heads, each token's cache is
        2 * 12 * 64 = 1536 floats per layer. With 4 KV heads, this drops to
        2 * 4 * 64 = 512 — a 3x reduction — with negligible quality loss
        (GQA paper, Llama 2/3, Mistral, Gemma all validate this).
    """
    def __init__(self, d_input, d_output, context_length, num_heads,
                 n_kv_heads=None, dropout=0.1, qkv_bias=False):
        super().__init__()
        assert d_output % num_heads == 0, "d_output must be divisible by num_heads"

        self.d_output = d_output
        self.num_heads = num_heads
        # Default to standard MHA if n_kv_heads not specified
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.head_dim = d_output // num_heads
        # Number of query heads per KV head — used for repeating KV heads
        # so that shapes align during the dot-product attention.
        self.n_rep = self.num_heads // self.n_kv_heads

        # Q projection: always projects to d_output (num_heads * head_dim).
        # K/V projections: project to n_kv_heads * head_dim (fewer heads = less cache).
        # GQA saves memory because K/V projections are smaller.
        self.W_query = nn.Linear(d_input, d_output, bias=qkv_bias)
        self.W_key = nn.Linear(d_input, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.W_value = nn.Linear(d_input, self.n_kv_heads * self.head_dim, bias=qkv_bias)

        self.out_proj = nn.Linear(d_output, d_output)
        self.dropout = nn.Dropout(dropout)

        # RoPE replaces learned absolute position embeddings.
        # The head_dim is the same for Q and K/V even though they have
        # different head counts — each KV head has the same dimensionality.
        self.rotary_emb = RotaryEmbedding(head_dim=self.head_dim, max_seq_len=8192)

        # Causal mask buffer: prevents token i from attending to token j>i.
        # We create a large enough buffer and slice into it for shorter sequences.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat KV heads to match the number of query heads for the
        dot-product attention computation.

        x: (batch, n_kv_heads, seq_len, head_dim)
        n_rep: number of query heads per KV head

        Returns: (batch, num_heads, seq_len, head_dim)

        Why repeat instead of broadcasting?
            torch.flash_attention and other fused kernels expect contiguous
            q, k, v with matching head counts. Broadcasting would require
            custom kernels or fallback to inefficient memory access patterns.
        """
        if n_rep == 1:
            return x
        batch, n_kv, seq_len, head_dim = x.shape
        # Expand: (b, n_kv, 1, sl, hd) -> (b, n_kv, n_rep, sl, hd) -> reshape to (b, n_kv*n_rep, sl, hd)
        return x[:, :, None, :, :].expand(batch, n_kv, n_rep, seq_len, head_dim).reshape(
            batch, n_kv * n_rep, seq_len, head_dim
        )

    def forward(self, x, past_keys=None, past_values=None):
        """
        x: input tensor (batch, num_tokens, d_in)
        past_keys: cached keys (batch, n_kv_heads, past_len, head_dim) or None
        past_values: cached values (batch, n_kv_heads, past_len, head_dim) or None

        Returns: (context_vec, updated_keys, updated_values)
        """
        b, num_tokens, d_in = x.shape

        # ---- Stage 1: Compute Q, K, V projections ----
        # Q: (b, num_tokens, d_output)    — full head count
        # K: (b, num_tokens, n_kv_heads * head_dim)  — reduced head count
        # V: (b, num_tokens, n_kv_heads * head_dim)  — reduced head count
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # ---- Stage 2: Reshape for multi-head ----
        # Split the last dimension into (n_heads, head_dim) or (n_kv_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.n_kv_heads, self.head_dim)
        values = values.view(b, num_tokens, self.n_kv_heads, self.head_dim)

        # Transpose to (b, n_heads, seq, head_dim) — standard attention layout
        queries = queries.transpose(1, 2)  # (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)        # (b, n_kv_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)    # (b, n_kv_heads, num_tokens, head_dim)

        # ---- Stage 3: Apply RoPE to Q and K ----
        # Determine absolute positions for the current input tokens:
        # - During prefill/training (past_keys=None): positions 0, 1, ..., num_tokens-1
        # - During incremental decode (past_keys is not None): 
        #   the new token is at position = past_len
        if past_keys is not None:
            # The new token(s) are at positions immediately after the cached prefix
            position_offset = past_keys.size(2)
            position_ids = torch.arange(
                position_offset, position_offset + num_tokens,
                device=x.device
            )
        else:
            # Full sequence from the start
            position_ids = torch.arange(num_tokens, device=x.device)

        # Rotate Q and K so that their dot-product encodes relative position
        queries, keys = self.rotary_emb(queries, keys, position_ids)

        # ---- Stage 4: KV cache ----
        # Concatenate already-rotated keys/values with cached prefix.
        # The cache contains keys rotated at their original positions, and the
        # new keys were just rotated at the correct positions above.
        if past_keys is not None and past_values is not None:
            keys = torch.cat([past_keys, keys], dim=2)
            values = torch.cat([past_values, values], dim=2)

        full_seq_len = keys.size(2)

        # Save compact K/V (n_kv_heads format) for caching.
        # The caller stores these and passes them back as past_keys/past_values
        # on the next decode step. Keeping n_kv_heads small saves cache memory.
        cache_keys = keys
        cache_values = values

        # ---- Stage 5: Repeat KV heads for GQA ----
        # Expand K, V from (b, n_kv_heads, seq, hd) to (b, num_heads, seq, hd)
        # so the matmul with Q works. Each KV head is repeated n_rep times.
        keys = self._repeat_kv(keys, self.n_rep)
        values = self._repeat_kv(values, self.n_rep)

        # ---- Stage 6: Scaled dot-product attention with causal masking ----
        attn_scores = queries @ keys.transpose(2, 3)  # (b, num_heads, seq, seq)

        # During training or first-pass (prefill): mask future tokens.
        # During autoregressive decode: only 1 token, nothing to mask.
        if past_keys is None and num_tokens > 1:
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ---- Stage 7: Weighted sum + output projection ----
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_output)
        context_vec = self.out_proj(context_vec)

        # Return updated keys/values in compact n_kv_heads format (before repeat)
        # so the cache stays small. The repeat is only done for the attention
        # computation and is not persisted between decode steps.
        return context_vec, cache_keys, cache_values
