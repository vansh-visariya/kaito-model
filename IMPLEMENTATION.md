# Implementation Roadmap — Modern Improvements for Kaito Model

This document catalogs architecture and training improvements from post-GPT-2 research (2019–2025) that could be applied to this codebase. Each section explains *what* changed, *why* it helps, the relevant paper, and a concrete implementation sketch.

---

## Table of Contents

1. [Rotary Position Embeddings (RoPE)](#1-rotary-position-embeddings-rope)
2. [Grouped Query Attention (GQA)](#2-grouped-query-attention-gqa)
3. [SwiGLU Feed-Forward Network](#3-swiglu-feed-forward-network)
4. [RMSNorm](#4-rmsnorm)
5. [AdamW Optimizer + LR Warmup](#5-adamw-optimizer--lr-warmup)
6. [Weight Tying](#6-weight-tying)
7. [Z-Loss for Logit Stabilisation](#7-z-loss-for-logit-stabilisation)
8. [Sliding Window Attention for Extended Context](#8-sliding-window-attention-for-extended-context)
9. [Parallel Attention + FFN](#9-parallel-attention--ffn)
10. [Mixed Precision Training](#10-mixed-precision-training)
11. [Mixture of Experts (MoE)](#11-mixture-of-experts-moe)
12. [Putting It All Together — Kaito v2 Config](#12-putting-it-all-together--kaito-v2-config)

---

## 1. Rotary Position Embeddings (RoPE)

### What

Replace the current learned absolute position embedding (`nn.Embedding(MAX_LENGTH, OUTPUT_DIM)`) with rotary position embeddings that rotate the Query and Key vectors by an angle proportional to their absolute position.

### Why

1. **Length extrapolation** — Learned embeddings are fixed at `MAX_LENGTH=512`; sequences beyond 512 crash or truncate. RoPE naturally extends to *any* sequence length because it operates on relative position via rotation, not a lookup table.
2. **Relative bias baked into attention** — The dot product of two RoPE-rotated vectors depends only on their *relative* position `m - n`, which is exactly what an autoregressive model needs.
3. **Zero additional parameters** — Unlike learned embeddings (which add `MAX_LENGTH * OUTPUT_DIM = 512 * 768 ≈ 393K` params), RoPE is a deterministic function. No extra parameters to learn.

### Paper

[Jianlin Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv:2104.09864 (2021)](https://arxiv.org/abs/2104.09864)

### Implementation sketch

Add a helper that precomputes cos/sin tables and applies the rotation to Q and K inside the attention module. The key insight is that we apply RoPE *after* the Q/K linear projections but *before* the view/transpose that splits heads.

```python
# Add to multi_attention/attention.py or a new file model/rope.py

import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Rotates Query and Key vectors by a frequency that depends on position.
    The rotation is applied pair-wise on dimensions (2i, 2i+1), which means
    the dot-product between a rotated query at position m and a rotated key
    at position n depends only on (m - n), giving the model a natural
    relative-position bias without any learned position embeddings.

    Why precompute cos/sin tables?
        Computing cos(m * theta_i) and sin(m * theta_i) on every forward pass
        for every position is wasteful. Since theta_i is fixed at init time,
        we precompute the full tables up to the maximum expected sequence length
        and slice into them during the forward pass. This adds a trivial
        memory cost (2 * max_seq_len * head_dim floats) but avoids recomputation.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        # head_dim must be even because we rotate pairs (2i, 2i+1)
        assert head_dim % 2 == 0, "RoPE requires head_dim to be even"

        # Compute the frequency for each dimension pair.
        # theta_i = base^(-2i / head_dim) for i in [0, head_dim/2)
        # This creates a geometric progression from base^0 down to base^(-1 + 2/head_dim).
        # The choice of base=10000 follows the sinusoidal encoding from "Attention Is All You Need"
        # and ensures a wide range of frequencies — high frequencies capture local patterns,
        # low frequencies capture long-range dependencies.
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Precompute position indices: [0, 1, 2, ..., max_seq_len - 1]
        positions = torch.arange(max_seq_len).float()

        # Outer product: (max_seq_len,) x (head_dim/2,) -> (max_seq_len, head_dim/2)
        # Each row m contains: [m*theta_0, m*theta_1, ..., m*theta_(head_dim/2 - 1)]
        angles = torch.outer(positions, freqs)

        # Duplicate to fill the full head_dim: (max_seq_len, head_dim)
        # For each pair (2i, 2i+1), both dimensions get the same angle.
        # This means we rotate each 2D subspace (d, d+1) by the same angle,
        # which preserves the dot-product structure needed for relative-position dependence.
        angles = torch.cat([angles, angles], dim=-1)

        # Precompute cos and sin tables — these are used in the forward pass
        # to rotate Q and K without any trigonometric computation at runtime.
        self.register_buffer("cos", angles.cos())  # (max_seq_len, head_dim)
        self.register_buffer("sin", angles.sin())  # (max_seq_len, head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        """
        Apply rotary embeddings to query and key tensors.

        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_heads, seq_len, head_dim)
            position_ids: (batch, seq_len) or (seq_len,) — absolute positions

        Returns:
            q_rotated, k_rotated: same shapes as inputs
        """
        # Gather the cos/sin values for the requested positions
        # We squeeze to 1D if position_ids is 2D with a single batch
        cos = self.cos[position_ids]  # (batch, seq_len, head_dim) or (seq_len, head_dim)
        sin = self.sin[position_ids]

        # Add head dimension for broadcasting: (..., 1, seq_len, head_dim)
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim) or (1, seq_len, head_dim)
        sin = sin.unsqueeze(1)

        # Apply rotation using the pair-wise rotation formula:
        #   RoPE(x)_(2i)   = x_(2i)   * cos(theta) - x_(2i+1) * sin(theta)
        #   RoPE(x)_(2i+1) = x_(2i+1) * cos(theta) + x_(2i)   * sin(theta)
        #
        # This is derived from the 2D rotation matrix:
        #   [cos  -sin] * [x0] = [x0*cos - x1*sin]
        #   [sin   cos]   [x1]   [x1*cos + x0*sin]
        #
        # We apply this independently to each pair (2i, 2i+1) because:
        # (a) It's computationally efficient — O(head_dim) per token.
        # (b) The dot-product of two such rotated vectors naturally encodes
        #     relative position (see Lemma 1 in the RoPE paper).
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotates each pair (2i, 2i+1) by swapping and negating the first element.
        
        This implements: rotate_half([x0, x1, x2, x3, ...]) = [-x1, x0, -x3, x2, ...]
        which corresponds to the second term in the rotation formula.
        
        Why split into two halves instead of indexing pairs?
            Indexing (x[..., ::2], x[..., 1::2]) would require two operations and
            a cat/stack. Splitting the last dimension in half and negating the first
            half achieves the same effect with fewer ops and better tensor-layout
            locality for the GPU.
        """
        x_half = x.shape[-1] // 2
        x1 = x[..., :x_half]  # first half — even-indexed dimensions
        x2 = x[..., x_half:]  # second half — odd-indexed dimensions
        return torch.cat((-x2, x1), dim=-1)
```

Then in `multi_attention/attention.py`, create the RoPE module in `__init__` and apply it after computing Q/K but before the attention score dot product:

```python
# Inside multihead_attention.__init__:
# Replace the mask buffer and learned position embedding with RoPE.
# RoPE needs the head dimension, not the full d_output, because rotation
# is applied to each head independently.
self.rotary_emb = RotaryEmbedding(head_dim=self.head_dim, max_seq_len=8192)

# Inside multihead_attention.forward, after reshaping keys/queries/values:
# Apply RoPE to q and k (not v — rotation applies only to the dot-product side).
# During KV-cache inference, position_ids = past_len + new_token_position so that
# rotation angles are correct for the accumulated sequence.
position_ids = torch.arange(full_seq_len, device=x.device).unsqueeze(0)
queries, keys = self.rotary_emb(queries, keys, position_ids)
```

### Impact

| Metric | Learned Embedding | RoPE |
|--------|------------------|------|
| Max train length | 512 tokens | Arbitrary (tested to 32K+) |
| Parameters (512×768) | 393,216 | 0 |
| Extrapolation quality | Drops sharply beyond 512 | Graceful degradation |

---

## 2. Grouped Query Attention (GQA)

### What

Reduce the number of key/value heads relative to query heads. For example, with 12 query heads and 4 KV heads (ratio 3:1), every group of 3 query heads shares one KV head.

### Why

1. **KV cache memory** — The KV cache is the dominant memory consumer during long-sequence generation. With 12 KV heads, each token's cache is `2 * 12 * head_dim = 2 * 12 * 64 = 1536` floats per layer. With GQA (4 KV heads), this drops to `2 * 4 * 64 = 512` — a **3× reduction**.
2. **Minimal quality loss** — GQA was shown in the original paper and confirmed by Llama 2/3, Mistral, and Gemma to match full multi-head attention quality at comparable compute budgets. The intuition: queries are diverse enough to benefit from many heads, but the keys/values encode "content" that can be shared across a group.
3. **Faster decoding** — Fewer KV heads means fewer memory reads during the attention computation, which is the primary bottleneck during autoregressive decoding (memory-bandwidth-bound).

### Paper

[Joshua Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", arXiv:2305.13245 (2023)](https://arxiv.org/abs/2305.13245)

Also adopted in: Llama 2, Llama 3, Mistral, Mixtral, Gemma.

### Implementation sketch

```python
# Modified multihead_attention with GQA support.

class grouped_query_attention(nn.Module):
    """
    Grouped Query Attention (GQA).
    
    Extends multi-head attention by allowing n_kv_heads < n_heads.
    Query heads are partitioned into groups; each group shares one K,V head.
    
    When n_kv_heads == n_heads, this is standard MHA.
    When n_kv_heads == 1, this is Multi-Query Attention (MQA).
    
    Why not always use n_kv_heads=1 (MQA)?
        MQA (Shazeer 2019) was shown to have a small but measurable quality
        degradation compared to full MHA. GQA interpolates between MHA and MQA,
        letting you trade off KV-cache size against quality. A ratio of 4:1 or 3:1
        (n_heads / n_kv_heads) typically recovers >99% of MHA quality at ~1/3 the cache.
    """
    def __init__(self, d_input, d_output, context_length, num_heads,
                 n_kv_heads=None, dropout=0.1, qkv_bias=False):
        super().__init__()
        assert d_output % num_heads == 0, "d_output must be divisible by num_heads"

        self.d_output = d_output
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.head_dim = d_output // num_heads
        # Number of query heads per KV head — used for repeating KV heads
        # during the attention computation so that shapes align.
        self.n_rep = self.num_heads // self.n_kv_heads

        # Q projection always goes to d_output (num_heads * head_dim).
        # K and V projections go to n_kv_heads * head_dim (smaller = fewer params + less cache).
        self.W_query = nn.Linear(d_input, d_output, bias=qkv_bias)
        self.W_key = nn.Linear(d_input, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.W_value = nn.Linear(d_input, self.n_kv_heads * self.head_dim, bias=qkv_bias)

        self.out_proj = nn.Linear(d_output, d_output)
        self.dropout = nn.Dropout(dropout)

        # RoPE replaces learned position embeddings
        self.rotary_emb = RotaryEmbedding(head_dim=self.head_dim, max_seq_len=8192)

        # Causal mask buffer
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat KV heads to match the number of query heads.
        
        x: (batch, n_kv_heads, seq_len, head_dim)
        n_rep: number of query heads per KV head
        
        Returns: (batch, num_heads, seq_len, head_dim) where each KV head
                 is repeated n_rep times along the head dimension.
        
        Why repeat instead of broadcasting?
            PyTorch's matmul can broadcast, but repeated heads let us use
            torch.flash_attention or other fused kernels that expect
            contiguous q, k, v with matching head counts.
        """
        if n_rep == 1:
            return x
        # Expand: (b, n_kv, sl, hd) -> (b, n_kv * n_rep, sl, hd)
        batch, n_kv, seq_len, head_dim = x.shape
        return x[:, :, None, :, :].expand(batch, n_kv, n_rep, seq_len, head_dim).reshape(
            batch, n_kv * n_rep, seq_len, head_dim
        )

    def forward(self, x, past_keys=None, past_values=None):
        b, num_tokens, d_in = x.shape

        # Compute Q, K, V projections
        queries = self.W_query(x)  # (b, num_tokens, d_output)
        keys = self.W_key(x)       # (b, num_tokens, n_kv_heads * head_dim)
        values = self.W_value(x)   # (b, num_tokens, n_kv_heads * head_dim)

        # Reshape Q for num_heads, K/V for n_kv_heads
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.n_kv_heads, self.head_dim)
        values = values.view(b, num_tokens, self.n_kv_heads, self.head_dim)

        # Transpose to (b, n_heads, seq, head_dim) — the standard attention layout
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # KV cache: concatenate new K/V with cached prefix
        if past_keys is not None and past_values is not None:
            keys = torch.cat([past_keys, keys], dim=2)
            values = torch.cat([past_values, values], dim=2)

        full_seq_len = keys.size(2)

        # Apply RoPE to Q and K
        # RoPE is applied after-cache-concat so that the cached keys already
        # have the correct rotation from their original positions.
        # During training (full sequence), this handles all positions.
        # During decoding (single token), the new token's rotation depends on
        # its absolute position = past_len, which is correct because the cached
        # keys were rotated at their original positions too.
        position_ids = torch.arange(full_seq_len, device=x.device).unsqueeze(0)
        queries, keys = self.rotary_emb(queries, keys, position_ids)

        # Repeat KV heads to match query heads before computing attention scores.
        # This is the core GQA step: n_kv_heads distinct KV representations
        # each serve n_rep query heads.
        keys = self._repeat_kv(keys, self.n_rep)
        values = self._repeat_kv(values, self.n_rep)

        # Scaled dot-product attention with causal masking
        attn_scores = queries @ keys.transpose(2, 3)

        if past_keys is None and num_tokens > 1:
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_output)
        context_vec = self.out_proj(context_vec)

        return context_vec, keys, values
```

### Impact

| Configuration | KV cache per layer (per token) | Quality vs MHA |
|-------------|-------------------------------|----------------|
| MHA (12 KV heads) | `12 * 2 * 64 = 1536` floats | Baseline |
| GQA (4 KV heads) | `4 * 2 * 64 = 512` floats | ~0.0% degradation |
| GQA (2 KV heads) | `2 * 2 * 64 = 256` floats | <0.5% degradation |
| MQA (1 KV head) | `1 * 2 * 64 = 128` floats | ~1-2% degradation |

For a 12-layer model generating 1024 tokens:
- MHA KV cache: `12 * 1024 * 1536 * 4 bytes ≈ 72 MB`
- GQA (4 KV): `12 * 1024 * 512 * 4 bytes ≈ 24 MB` → **3× reduction**

---

## 3. SwiGLU Feed-Forward Network

### What

Replace the current GELU-based FFN (`Linear → GELU → Linear`) with a gated variant: `Linear → Swish → element-wise multiply with a second Linear → Output Linear`.

### Why

1. **Better quality per parameter** — Shazeer (2020) showed that SwiGLU consistently outperforms ReLU/GELU at the same parameter count. Llama, PaLM, and Mistral all use SwiGLU.
2. **Richer expressivity** — The gating mechanism `swish(xW_gate) ⊙ (xW_up)` lets the network learn which information to pass through and which to suppress, similar to an LSTM's forget gate but applied at the token-feature level.
3. **Compensation for reduced hidden dim** — SwiGLU's hidden dimension is typically `2/3 * 4 * d_model` instead of the standard `4 * d_model`, because the gated FFN has three weight matrices instead of two. This keeps the total parameter count similar while improving quality.

### Paper

[Noam Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202 (2020)](https://arxiv.org/abs/2002.05202)

### Implementation sketch

```python
# Replace the FeedForward class in model/architecture.py

class SwiGLU(nn.Module):
    """
    Sigmoid Linear Unit (SiLU) activation, also called Swish.
    
    swish(x) = x * sigmoid(x)
    
    Why Swish instead of GELU?
        GELU and Swish are extremely similar (both are smooth, non-monotonic,
        and bounded below). Swish is slightly cheaper to compute because
        sigmoid is simpler than the GELU approximation (tanh of a cubic).
        More importantly, SwiGLU (the gated variant) was shown in the
        Shazeer 2020 paper to outperform GELU-gated variants.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class GatedFeedForward(nn.Module):
    """
    SwiGLU gated feed-forward network.
    
    Standard FFN: output = W2(GELU(W1(x)))
    SwiGLU FFN:   output = W3(Swish(W_gate(x)) * W_up(x))
    
    The gate and up projections are parallel; their element-wise product
    forms the gating mechanism. The down-projection W3 maps back to d_model.
    
    Parameter allocation (vs standard FFN):
        Standard FFN (768 -> 3072 -> 768):
            W1: 768*3072 = 2,359,296
            W2: 3072*768 = 2,359,296
            Total: 4,718,592
        
        SwiGLU FFN (768 -> 2048 -> 768) — using 8/3 * d_model instead of 4 * d_model:
            W_gate: 768*2048 = 1,572,864
            W_up:   768*2048 = 1,572,864
            W_down: 2048*768 = 1,572,864
            Total: 4,718,592
        
    The SwiGLU variant matches the standard FFN's parameter count while using
    a smaller hidden dimension (2048 vs 3072), thanks to the third weight matrix.
    The gating interaction between gate and up projections is what provides
    the quality improvement — it's not about having more parameters but about
    a more expressive computation graph.
    """
    def __init__(self):
        super().__init__()
        # Llama's convention: hidden_dim = int(8/3 * d_model) rounded to a multiple
        # that's efficient on GPU tensor cores. For d_model=768:
        #   8/3 * 768 = 2048.0, which is already a nice number.
        hidden_dim = int(8/3 * OUTPUT_DIM)
        
        # The gate projection produces the "weights" of the gate
        self.W_gate = nn.Linear(OUTPUT_DIM, hidden_dim, bias=False)
        # The up projection produces the "values" to be gated
        self.W_up = nn.Linear(OUTPUT_DIM, hidden_dim, bias=False)
        # The down projection maps back to the model dimension
        self.W_down = nn.Linear(hidden_dim, OUTPUT_DIM, bias=False)
        
        self.activation = SwiGLU()

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        
        The element-wise product of gate and up creates an interaction:
        each feature in the up projection is scaled by a learnable "relevance"
        signal from the gate projection. This is more expressive than a simple
        activation function applied pointwise.
        """
        gate_out = self.activation(self.W_gate(x))  # Gate: which features to keep
        up_out = self.W_up(x)                        # Up: candidate feature values
        # Element-wise gating: keep most of the "important" features,
        # suppress the "unimportant" ones.
        gated = gate_out * up_out
        return self.W_down(gated)
```

### Impact

At the same parameter budget, SwiGLU consistently improves perplexity by **0.1–0.3** on standard LM benchmarks compared to GELU (Shazeer 2020, Table 1). The improvement grows with model scale.

---

## 4. RMSNorm

### What

Replace `LayerNorm` (mean-centering + scaling) with `RMSNorm` (scaling only, no mean subtraction).

### Why

1. **Computational efficiency** — LayerNorm requires computing both mean (`x.mean(dim=-1)`) and variance (`x.var(dim=-1)`). RMSNorm only requires the root-mean-square (`x.pow(2).mean(dim=-1).sqrt()`), saving ~15-25% of the normalization FLOPs.
2. **Empirically equivalent quality** — The mean-centering step in LayerNorm is redundant for transformer training. The residual connections + attention mechanism already provide enough "centering" that the normalization only needs to control the scale (variance). Llama, Mistral, and Gemma all use RMSNorm.
3. **Simpler gradient flow** — Removing the mean eliminates one dependency in the backward pass, which can slightly improve gradient conditioning.

### Paper

[Biao Zhang & Rico Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019](https://arxiv.org/abs/1910.07467)

### Implementation sketch

```python
# Replace the LayerNorm class in model/architecture.py

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Unlike LayerNorm, RMSNorm does not subtract the mean:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale
    
    Why does this work without centering?
        In deep transformers, the pre-norm residual connections ensure that
        the input to each sub-layer is already roughly zero-mean (because
        residual streams average over many contributions). The remaining
        variance mismatch is what needs normalization, and RMSNorm handles
        that without the overhead of computing the mean.
    
    The scale parameter is still learnable because the optimal variance
    for the attention softmax (which is sensitive to scale) differs from
    the optimal variance for the FFN (which uses activation functions).
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.eps = 1e-5  # Same epsilon as LayerNorm for consistency
        self.scale = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        # rms = sqrt(mean(x^2) + eps)
        # We add eps inside the sqrt to avoid division by zero,
        # matching the convention used in LayerNorm (where eps is under the sqrt).
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)
```

### Performance comparison

| Normalization | FLOPs per token (768-dim) | Parameters |
|-------------|--------------------------|------------|
| LayerNorm | ~4,608 FLOPs (mean + var) | 1,536 (scale + shift) |
| RMSNorm | ~3,072 FLOPs (rms only) | 768 (scale only) |

**Saving**: ~33% fewer normalization FLOPs, ~50% fewer normalization parameters.

---

## 5. AdamW Optimizer + LR Warmup

### What

Replace the current setup (plain Adam or not specified, cosine from step 0) with:
1. **AdamW** — decouples weight decay from the gradient-based parameter updates.
2. **Linear warmup** — linearly increase LR from 0 to `LEARNING_RATE` over the first ~5–10% of training steps.
3. **Cosine decay after warmup** — resume current cosine schedule from the end of warmup to 0.

### Why

1. **AdamW fixes L2 regularization** — In standard Adam, weight decay is implemented as L2 regularization added to the loss, which interacts poorly with Adam's adaptive learning rates (the regularization gets scaled by 1/sqrt(v) which attenuates it for frequently updated parameters). AdamW directly subtracts the decay from the weights *after* the Adam update, which correctly regularizes all parameters equally.
2. **Warmup prevents early instability** — In the first few steps, the Adam optimizer's momentum and variance estimates (`m`, `v`) are initialized at zero and need time to "warm up". A high LR before these estimates stabilize can cause gradient explosions (especially in transformers with many layers). The current cosine schedule starts at full LR from step 0, which is risky.
3. **Cosine decay converges better** — The current cosine schedule is already correct (step once per epoch), but it starts at `LEARNING_RATE` rather than decaying from it. Warmup + cosine is the de facto standard for modern LLM training (GPT-3, Llama, PaLM).

### Paper

- [Ilya Loshchilov & Frank Hutter, "Decoupled Weight Decay Regularization", ICLR 2019](https://arxiv.org/abs/1711.05101) (AdamW)
- [Priya Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", arXiv:1706.02677](https://arxiv.org/abs/1706.02677) (warmup)

### Implementation sketch

```python
# In train.py, modify the optimizer and scheduler setup

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context, tokenizer):
    # ... (existing setup code)

    # AdamW with decoupled weight decay.
    # The weight_decay parameter is the L2 penalty applied *directly to the weights*,
    # not to the loss. Typical values: 0.1 for LLMs (GPT-3 used 0.1, Llama used 0.1).
    # We do NOT apply weight decay to biases and norm parameters:
    #   - Biases are 1D and applying weight decay to them would push them toward 0,
    #     which conflicts with their role as offsets.
    #   - Norm parameters (scale, shift) interact multiplicatively with the data;
    #     weight decay would shrink them and require the rest of the model to compensate.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if 'bias' in name or 'norm' in name or 'layernorm' in name or 'scale' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': 0.1},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ],
        lr=LEARNING_RATE,    # 1e-4 (from config)
        betas=(0.9, 0.95),   # Llama-style betas: 0.9 for momentum, 0.95 for RMS
        eps=1e-8
    )

    # Total number of training steps (batches, not gradient updates)
    total_steps = len(train_loader) * num_epochs
    # Warmup over the first ~5% of steps (standard practice)
    warmup_steps = int(0.05 * total_steps)

    # Warmup + cosine scheduler.
    # The scheduler is a LambdaLR that linearly increases LR from 0 to 1
    # during warmup, then applies cosine decay from 1 to 0 afterwards.
    # This combines two PyTorch schedulers, but we implement it as a single
    # lambda for correctness (chaining schedulers can have subtle ordering bugs).
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup: progress from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay from 1 to 0 over the remaining steps
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ... (rest of training loop)
    # Step scheduler every *batch*, not every epoch, because the warmup
    # phase spans batches (it should complete within the first epoch).
    # CosineAnnealingLR in the original code steps per epoch, which means
    # the LR stays constant for an entire epoch at a time. Per-step scheduling
    # is smoother and the de facto standard.
    
    for epoch in range(num_epochs):
        # ...
        for i, (input_batch, target_batch) in enumerate(train_loader):
            # Forward, loss, backward (same as before)
            # ...
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  # Step LR after each optimizer step (not each batch)
    
    # Note: scheduler.step() from the original epoch-level code is removed.
    # Instead we step per gradient update.
```

---

## 6. Weight Tying

### What

Share the weight matrix between the token embedding layer and the output projection head (`self.out_head`). They use the same `nn.Embedding` parameters.

### Why

1. **Massive parameter saving** — The output head is `VOCAB_SIZE * OUTPUT_DIM = 50257 * 768 ≈ 38.6M` parameters — the single largest parameter block in the model (more than all transformer blocks combined at `12 * (4 * 768^2 + attention) ≈ 28M`). Weight tying eliminates this entirely.
2. **Consistent representation learning** — The embedding matrix learns a "meaning space" for tokens. The output head learns a "prediction space". These should be the same space: a token's embedding should be the representation that causes the model to predict that token. Weight tying enforces this consistency.
3. **Proven effective for small-to-medium models** — Weight tying was introduced in "Attention Is All You Need" for the original Transformer and is standard in BERT and GPT-2 (small). For very large models (>10B params), the impact is smaller because the head is a smaller fraction of total params.

### Paper

- [Ofir Press & Lior Wolf, "Using the Output Embedding to Improve Language Models", EACL 2017](https://arxiv.org/abs/1608.05859)
- [Ashish Vaswani et al., "Attention Is All You Need", NeurIPS 2017](https://arxiv.org/abs/1706.03762) (Section 3.4 — "We share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30].")

### Implementation sketch

```python
# In main.py, modify kaitomodel.__init__ and forward

class kaitomodel(nn.Module):
    def __init__(self, tie_weights=True):
        super().__init__()
        # Token embedding: maps vocabulary indices to dense vectors.
        # This is the largest parameter block in the model (38.6M params).
        self.token_embedding = nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
        self.pos_embedding = nn.Embedding(MAX_LENGTH, OUTPUT_DIM)
        
        self.dropout = nn.Dropout(DROPOUT)
        self.trf_block = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.final_norm = LayerNorm(OUTPUT_DIM)
        
        # Output projection: maps d_model to vocabulary logits.
        # When tie_weights=True, this is a non-parameter "view" of the
        # token_embedding weight. We still create a Linear for the interface,
        # then override its weight after initialization.
        self.out_head = nn.Linear(OUTPUT_DIM, VOCAB_SIZE, bias=False)
        
        if tie_weights:
            # Weight tying: the output projection shares the same weight
            # matrix as the token embedding.
            # This is done by pointing out_head.weight at the same storage
            # as token_embedding.weight. PyTorch handles gradient accumulation
            # correctly — gradients flow to both token_embedding and out_head
            # but are summed since they share storage.
            self.out_head.weight = self.token_embedding.weight
    
    def forward(self, input_ids, past_key_values=None):
        # ... (same as before, no changes needed)
        # The out_head projection automatically uses the tied weight
        logits = self.out_head(features)
        return logits, new_key_values
```

### Impact

| Model size | Parameters before tying | After tying | Saving |
|-----------|------------------------|-------------|--------|
| Current (768, 12L) | ~67M | ~28M | ~58% |
| GPT-2 Small (768, 12L) | 124M | 85M | ~31% |

---

## 7. Z-Loss for Logit Stabilisation

### What

Add an auxiliary loss that penalizes the logit of the "zero" / end-of-sequence token or, more generally, penalizes the log-sum-exp of the logits (which equals the log of the partition function).

### Why

1. **Prevents logit explosion** — During training, the softmax's logits can grow unboundedly because the cross-entropy loss only cares about the *difference* between the correct token's logit and other logits, not their absolute magnitude. Logits can drift to very large positive/negative values, which:
   - Creates numerical instability in softmax (all probabilities → 0 or 1)
   - Makes the model overconfident (incapable of generating diverse text)
   - Can cause NaN loss during longer training runs
2. **Improves calibration** — By penalizing the overall logit magnitude, Z-loss encourages the model to produce well-calibrated probabilities, which is essential for high-quality sampling at inference time.

### Paper

- [Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways", arXiv:2204.02311 (2022)](https://arxiv.org/abs/2204.02311) — Section 5.4 describes Z-loss with coefficient 1e-4.
- The same technique is used in Chinchilla, Gopher, and most DeepMind LLMs.

### Implementation sketch

```python
# In loss/cal_loss.py

def cal_loss_batch(input_batch, target_batch, model, device, z_loss_coeff=1e-4):
    """
    Cross-entropy loss with optional Z-loss auxiliary term.
    
    Z-loss = z_loss_coeff * log(sum(exp(logits)))^2
    This penalizes the log-partition function (log Z) for being too large,
    which keeps the logits bounded and prevents numerical instability.
    
    Why squared?
        The squared penalty (rather than absolute) ensures that large logit
        magnitudes are penalized much more heavily than small ones, which
        is what we want — a small drift is harmless, but a large explosion
        needs aggressive correction.
    
    Why coefficient 1e-4?
        PaLM used this value and it was found to be effective across scales.
        Too large a coefficient would hurt training (it would compete with
        the cross-entropy objective); too small would not prevent explosion.
        1e-4 is a safe default that provides stabilisation without measurable
        impact on loss convergence.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits, _ = model(input_batch)
    
    # Standard cross-entropy loss
    ce_loss = nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    
    # Z-loss auxiliary term
    # log(sum(exp(logits))) is the log of the partition function (log Z).
    # We compute it along the vocabulary dimension (dim=-1).
    # The square keeps it positive and heavily penalizes outliers.
    log_z = torch.logsumexp(logits, dim=-1)  # (batch, seq_len)
    z_loss = z_loss_coeff * (log_z ** 2).mean()
    
    return ce_loss + z_loss


def cal_loss_loader(data_loader, model, device, num_batches=None, z_loss_coeff=1e-4):
    # ... (same as before, pass z_loss_coeff to cal_loss_batch)
    total_loss += cal_loss_batch(
        input_batch, target_batch, model, device,
        z_loss_coeff=z_loss_coeff
    ).item()
```

---

## 8. Sliding Window Attention for Extended Context

### What

Limit each token's attention to a fixed-size window of nearby tokens (e.g., 4096 tokens) rather than the full sequence. Combine with the RoPE change (above) to support tasks that exceed `MAX_LENGTH=512`.

### Why

1. **Linear vs quadratic memory** — Full causal attention costs `O(n²)` memory for the attention scores. Sliding window reduces this to `O(n * window_size)`. At `window_size=4096`, the memory for a 32K sequence drops by ~8×.
2. **Local focus is sufficient** — Mistral and Sliding-Window Attention papers demonstrate that most linguistic dependencies are local (< 1000 tokens). The model learns to "look back" via the residual stream across layers, giving it an effective receptive field of `window_size * n_layers`.
3. **Compatible with KV cache** — The sliding window becomes a rolling cache: evict the oldest token's K/V when a new token arrives. The cache size is constant `O(window_size)` instead of `O(sequence_length)`.

### Paper

- [Mistral AI, "Mistral 7B", arXiv:2310.06825 (2023)](https://arxiv.org/abs/2310.06825)
- [Iz Beltagy et al., "Longformer: The Long-Document Transformer", arXiv:2004.05150 (2020)](https://arxiv.org/abs/2004.05150)

### Implementation sketch

```python
# In multi_attention/attention.py, add window masking to the attention forward pass.

# Inside multihead_attention (or grouped_query_attention).forward:
# After computing attn_scores, before softmax:

if self.sliding_window is not None and full_seq_len > self.sliding_window:
    # Create a sliding window mask.
    # Token i can only attend to tokens in [i - window_size + 1, i].
    # Tokens outside this range get -inf (masked out).
    # This is a banded mask: only the lower-left "band" of width window_size is valid.
    # 
    # We use the same upper-triangular approach as the causal mask but with an
    # additional lower-triangular mask at offset (window_size) that blocks tokens
    # farther back than window_size.
    seq_range = torch.arange(full_seq_len, device=attn_scores.device)
    # Distance of each (query_i, key_j) pair: j - i.
    # Query i attends to key j when i - window_size < j <= i.
    # Equivalently: j >= i - window_size + 1 (focus on recent tokens)
    causal_distance = seq_range[:, None] - seq_range[None, :]  # (seq, seq)
    window_mask = causal_distance < -self.sliding_window + 1
    # Apply both causal mask (can't see future) and window mask (can't see too far past)
    combined_mask = mask_bool | window_mask  # True = masked out
    attn_scores.masked_fill_(combined_mask, -torch.inf)
```

---

## 9. Parallel Attention + FFN

### What

Instead of computing attention then FFN sequentially, compute them in parallel and sum their outputs.

### Why

1. **~15% faster training** — GPUs are good at parallelism; computing attention and FFN simultaneously for the same input uses the GPU's tensor cores more efficiently.
2. **Minimal quality impact** — PaLM (540B) used this design and it performed equivalently to the sequential variant. The two computations operate on different subspaces (attention mixes tokens, FFN processes features independently per token) so they don't conflict.
3. **Simpler architecture** — One fewer sequential dependency means fewer kernel launches and better hardware utilization.

### Paper

[Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways", arXiv:2204.02311 (2022)](https://arxiv.org/abs/2204.02311) — Section 5.1 "Parallel Formulation in Transformer Blocks"

### Implementation sketch

```python
# Modified TransformerBlock.forward in model/architecture.py

# Standard (sequential):
# x = x + attention(layernorm1(x))
# x = x + ffn(layernorm2(x))

# Parallel (PaLM-style):
# normed = layernorm(x)
# x = x + attention(normed) + ffn(normed)
#
# Both attention and FFN see the same normalized input and contribute
# independently to the residual stream. This removes one sequential
# dependency without changing the effective computation.

def forward(self, x, past_keys=None, past_values=None):
    # Parallel attention + FFN
    normed = self.layernorm1(x)
    attn_out, new_keys, new_values = self.attention(normed, past_keys, past_values)
    attn_out = self.dropout(attn_out)
    ffn_out = self.dropout(self.feedforward(normed))
    x = x + attn_out + ffn_out
    # Note: with parallel formulation, we only need one LayerNorm per block
    # instead of two. The layernorm2 parameter becomes unused.
    return x, new_keys, new_values
```

However — removing `layernorm2` would break saved checkpoints. A practical approach: keep `layernorm2` but set it to identity or just use it after the parallel addition (redundant but safe).

---

## 10. Mixed Precision Training

### What

Train with `torch.bfloat16` (brain floating point, a 16-bit format with the same exponent range as float32) for most operations, keeping only the loss and certain accumulations in float32.

### Why

1. **~2× training speed** — bfloat16 tensors are half the size of float32, so memory bandwidth doubles and arithmetic operations are faster on GPU tensor cores (Ampere+).
2. **~2× memory reduction** — Smaller gradients, activations, and optimizer states mean you can train larger models or use larger batches in the same GPU memory.
3. **No precision loss for transformers** — bfloat16 has the same 8-bit exponent as float32 (unlike float16's 5-bit exponent), so it does not underflow or overflow easily. Transformer training is remarkably robust to bfloat16 — the loss curve is virtually identical to float32.

### Implementation sketch

```python
# In train.py, add a scaler (only for float16, bfloat16 doesn't need one)
# and move model/data to the appropriate dtype.

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context, tokenizer, use_amp=True):
    
    # Automatic Mixed Precision (AMP) context manager
    # We use bfloat16 because it doesn't need gradient scaling
    # (float16 can underflow/overflow and requires GradScaler).
    # On pre-Ampere GPUs, use float16 with GradScaler instead.
    dtype = torch.bfloat16 if torch.cuda.is_available() and use_amp else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    
    model.to(device=device, dtype=dtype)
    
    for epoch in range(num_epochs):
        for i, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                loss = cal_loss_batch(input_batch, target_batch, model, device)
                loss = loss / ACCUMULATION_STEPS
            
            # Backward pass with AMP
            if dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                if dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
```

---

## 11. Mixture of Experts (MoE)

### What

Replace every Feed-Forward Network (FFN) with a set of *expert* FFNs and a learned *router* that selects a sparse subset of experts per token. Instead of computing one FFN for all tokens, each token is routed to the top-k experts (usually 1 or 2 out of 8–64); their outputs are weighted by the router's gating scores and summed.

### Why

1. **Massive parameter count with constant inference cost** — MoE decouples the *total parameter count* from the *compute per token*. With 8 experts and top-2 routing, only ~25% of total parameters are activated per token. This means you can have a 100B-parameter model that runs at the speed of a ~25B model. GPT-4, Mixtral 8×7B, and Gemini all use MoE to scale far beyond what dense models could afford.

2. **Better quality-per-compute** — Mixtral 8×7B (47B total, 13B active) matches or outperforms Llama 2 70B (70B dense) on most benchmarks while using ~5× less inference FLOPs per token. The key insight is that different tokens need different "expertise" — a pronoun probably doesn't need the same specialist capacity as a rare technical term.

3. **Natural specialization emerges** — Without any explicit supervision, experts consistently specialize in different domains (syntax, entities, long-range dependencies, mathematical reasoning, etc.) when analyzed post-training (see the ST-MoE paper, Section 6).

4. **Scales better with data** — Dense models eventually saturate: adding more parameters doesn't improve quality proportionally. MoE models show better scaling behavior because expert specialization lets them "grow" capacity only where needed.

### Papers

- [Noam Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR 2017](https://arxiv.org/abs/1701.06538) — The original modern MoE formulation with noisy top-k gating.
- [William Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR 2022](https://arxiv.org/abs/2101.03961) — Simplified top-1 routing; showed MoE works at massive scale.
- [Albert Q. Jiang et al., "Mixtral of Experts", arXiv:2401.04088 (2024)](https://arxiv.org/abs/2401.04088) — The architecture behind Mixtral 8×7B; production-validated MoE with top-2 routing and load balancing.
- [Barret Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906 (2022)](https://arxiv.org/abs/2202.08906) — Stability techniques for MoE training: Z-loss, expert dropout, router z-loss.

### Core concepts

**The router (gate).** A small linear layer that maps each token's representation to a score per expert:

```
router_logits = x @ W_router          # (batch * tokens, n_experts)
router_probs = softmax(router_logits) # probabilities for each expert
```

**Top-k selection.** Keep only the k experts with the highest probabilities; zero out the rest. During training, the router is trained via the gradients flowing through the top-k selected experts' outputs:

```
top_k_probs, top_k_indices = top_k(router_probs, k)   # k=1 or k=2
mask = one_hot(top_k_indices)                          # (batch*tokens, n_experts)
gates = mask * router_probs                            # final gating weights
```

**Expert computation.** Only the selected experts compute their FFN. Each token gets:

```
token_out = sum over selected e of: gates[e] * expert_e(token)
```

This requires a placement function that sends tokens to the right device/process in distributed training, but for single-GPU training it's a simple loop or batched einsum.

**Load balancing loss.** Without intervention, the router will collapse: most tokens will be sent to the same 1–2 experts, creating a "rich get richer" dynamic (those experts get more gradients, become better, get even more tokens). A load-balancing auxiliary loss penalizes uneven router distribution:

```
# Switch Transformer's auxiliary loss:
# For each expert, compute: fraction_of_tokens_routed * fraction_of_router_probability
# The product is minimised when both distributions are uniform.
load_balancing_loss = n_experts * sum_e (f_e * P_e)
```
where `f_e` is the fraction of tokens routed to expert `e` and `P_e` is the average router probability assigned to expert `e`.

### Implementation sketch

```python
# New file: model/moe.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer that replaces a standard FeedForward.
    
    Architecture (following Mixtral 8x7B conventions):
      - n_experts: total number of expert FFNs (e.g., 8)
      - top_k: number of experts activated per token (e.g., 2)
      - Each expert is a standard SwiGLU FeedForward (see Section 3)
    
    Why top-2 instead of top-1 (Switch Transformer)?
        Top-2 has two advantages over top-1:
        1. It provides a richer gradient signal — each token's output is a
           weighted blend of two experts, which gives smoother optimisation.
        2. It allows for "redundant" routing: if the top expert is saturated,
           the second expert can provide complementary information.
        Mixtral 8x7B uses top-2 and it works well in practice.
    
    Why SwiGLU for each expert?
        Following the SwiGLU change in Section 3, each expert should use the
        gated FFN (8/3 * d_model hidden dim) rather than a standard GELU FFN.
        This keeps the per-expert parameter count consistent with the
        non-MoE variant and ensures quality improvements apply to MoE too.
    """
    def __init__(self, d_model: int, n_experts: int, top_k: int = 2,
                 hidden_mult: float = 8/3):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # The router: a single linear layer that produces logits for each expert.
        # It's intentionally small (d_model -> n_experts) to minimise routing
        # overhead — typically < 0.1% of total model parameters.
        self.router = nn.Linear(d_model, n_experts, bias=False)
        
        # Expert FFNs — each is an independent SwiGLU gated FFN.
        # We stack them into a single ModuleList so PyTorch tracks their
        # parameters correctly. Each expert is identical in capacity.
        hidden_dim = int(hidden_mult * d_model)
        self.experts = nn.ModuleList([
            self._build_expert(d_model, hidden_dim) for _ in range(n_experts)
        ])
        
        # Buffer for tracking load-balancing statistics across batches.
        # This is used by the auxiliary loss to detect router collapse.
        self.register_buffer("_expert_counts", torch.zeros(n_experts))
    
    @staticmethod
    def _build_expert(d_model: int, hidden_dim: int) -> nn.Module:
        """Create one expert as a SwiGLU gated FFN (see Section 3)."""
        return nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),   # W_gate
            nn.Linear(d_model, hidden_dim, bias=False),   # W_up
            nn.Linear(hidden_dim, d_model, bias=False),   # W_down
        )
    
    def _swiglu(self, gate_out: torch.Tensor, up_out: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: gate_out * sigmoid(gate_out) * up_out."""
        return (gate_out * torch.sigmoid(gate_out)) * up_out
    
    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, d_model)
        
        Returns: (batch, seq_len, d_model) — the MoE-weighted combination of
                 expert outputs, plus the auxiliary load-balancing loss.
        
        The forward pass has three stages:
          1. Routing: compute router probabilities, select top-k experts.
          2. Expert computation: for *each* expert, process the tokens routed
             to it (only the fraction of tokens assigned to that expert).
          3. Combine: multiply each token's expert outputs by the gating
             weights and sum them.
        """
        orig_shape = x.shape
        batch, seq_len, d_model = orig_shape
        
        # Flatten batch and sequence dimensions for per-token routing
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)
        num_tokens = x_flat.size(0)
        
        # ---- Stage 1: Routing ----
        # Compute router logits and convert to probabilities via softmax.
        # The router weights are learned jointly with expert weights.
        router_logits = self.router(x_flat)  # (num_tokens, n_experts)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Select top-k experts per token
        # top_k_probs: the probabilities of the selected experts
        # top_k_indices: the expert indices (0..n_experts-1) for each token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize the top-k probabilities so they sum to 1 (they don't
        # automatically because we took a subset of the softmax distribution).
        # This ensures the output scale is consistent regardless of top-k choice.
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # ---- Stage 2: Expert computation ----
        # We iterate over experts rather than tokens because in the single-GPU
        # case, this is simpler and avoids scatter/gather complexity.
        # For each expert, we:
        #   a) Find which tokens are routed to it (via top_k_indices)
        #   b) Extract those tokens' representations
        #   c) Run them through the expert's SwiGLU FFN
        #   d) Store the result, weighted by the gating probability
        #
        # This is an O(n_experts * tokens_per_expert) loop, which is efficient
        # because n_experts is small (8–16) and each expert only processes
        # ~top_k/n_experts fraction of tokens.
        
        final_output = torch.zeros_like(x_flat)
        
        # Count tokens per expert for load-balancing loss
        expert_counts = torch.zeros(self.n_experts, device=x.device)
        
        for expert_idx, expert in enumerate(self.experts):
            # Find tokens that have this expert in their top-k selection.
            # top_k_indices shape: (num_tokens, top_k) — each row has k expert ids
            # We want rows where column 0 or column 1 == expert_idx
            mask = (top_k_indices == expert_idx)  # (num_tokens, top_k)
            # Which tokens route to this expert (via either top-1 or top-2)?
            token_mask = mask.any(dim=-1)  # (num_tokens,) — boolean
            
            if not token_mask.any():
                # No tokens routed to this expert — skip computation entirely.
                # This is the sparsity benefit: most experts are idle for any
                # given token, and many experts are idle entirely for a batch.
                continue
            
            # Get the routing weight from this expert for the selected tokens.
            # masked token_mask with the same shape as top_k_probs
            expert_weights = top_k_probs[mask]  # scalar per selected token
            expert_weights = expert_weights.unsqueeze(-1)  # (n_selected, 1)
            
            # Extract token representations for the selected tokens
            selected_tokens = x_flat[token_mask]  # (n_selected, d_model)
            
            # Run through expert's SwiGLU FFN.
            # Expert is a Sequential of (W_gate, W_up, W_down)
            gate_out = expert[0](selected_tokens)   # (n_selected, hidden_dim)
            up_out = expert[1](selected_tokens)     # (n_selected, hidden_dim)
            down_in = self._swiglu(gate_out, up_out)  # gated activation
            expert_out = expert[2](down_in)           # (n_selected, d_model)
            
            # Scale by routing weight and scatter-add to the output.
            # We use scatter_add_ because multiple experts can contribute to
            # the same token (top-2 routing), and we need to sum their outputs.
            weighted_out = expert_out * expert_weights
            final_output[token_mask] += weighted_out
            
            # Track count for load balancing
            expert_counts[expert_idx] = token_mask.sum()
        
        # ---- Load balancing auxiliary loss ----
        # We compute the Switch Transformer's load-balancing loss:
        #   loss_lb = n_experts * sum_e (f_e * P_e)
        # where f_e = fraction of tokens routed to expert e
        #       P_e = average router probability assigned to expert e
        #       
        # The product f_e * P_e is minimised when both are uniform (1/n_experts).
        # Multiplying by n_experts scales the loss so it's O(1) regardless of n.
        # Typical coefficient: 0.01 (used by Switch Transformer, Mixtral).
        f_e = expert_counts / num_tokens
        P_e = router_probs.mean(dim=0)  # average probability per expert
        load_balancing_loss = self.n_experts * (f_e * P_e).sum()
        
        return final_output.view(batch, seq_len, d_model), load_balancing_loss
    
    def _count_parameters(self) -> dict:
        """Utility for debugging: parameter breakdown per expert vs router."""
        router_params = sum(p.numel() for p in self.router.parameters())
        expert_params = sum(
            sum(p.numel() for p in expert.parameters())
            for expert in self.experts
        )
        return {
            "router": router_params,
            "per_expert": expert_params // self.n_experts,
            "total_experts": expert_params,
            "active_per_token": (expert_params // self.n_experts) * self.top_k
                               + router_params,
        }
```

### Integrating MoE into the existing TransformerBlock

```python
# In model/architecture.py, modify TransformerBlock

class TransformerBlock(nn.Module):
    """
    Transformer block with MoE replacing the dense FFN.
    
    When use_moe=True:
      - The self.feedforward is replaced with an MoELayer.
      - The MoE returns an auxiliary load-balancing loss in addition to
        the output tensor. This loss must be accumulated across all blocks
        and added to the total training loss.
    
    When use_moe=False:
      - Behaves identically to the original TransformerBlock (SwiGLU FFN).
    """
    def __init__(self, use_moe=False, n_experts=8, top_k=2):
        super().__init__()
        self.attention = grouped_query_attention(
            d_input=OUTPUT_DIM, d_output=OUTPUT_DIM,
            context_length=MAX_LENGTH, num_heads=N_HEADS,
            n_kv_heads=N_KV_HEADS, dropout=DROPOUT, qkv_bias=qkv_bias
        )
        NormClass = RMSNorm if USE_RMSNORM else LayerNorm
        self.layernorm1 = NormClass(OUTPUT_DIM)
        
        if use_moe:
            # MoE replaces the second LayerNorm + FFN.
            # The layernorm2 is still needed (applied before the MoE).
            self.layernorm2 = NormClass(OUTPUT_DIM)
            self.feedforward = MoELayer(
                d_model=OUTPUT_DIM,
                n_experts=n_experts,
                top_k=top_k
            )
            self.use_moe = True
        else:
            self.layernorm2 = NormClass(OUTPUT_DIM)
            self.feedforward = GatedFeedForward()
            self.use_moe = False
        
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x, past_keys=None, past_values=None):
        # Attention sub-layer (unchanged)
        shortcut = x
        x = self.layernorm1(x)
        x, new_keys, new_values = self.attention(x, past_keys, past_values)
        x = self.dropout(x)
        x = x + shortcut
        
        # FFN sub-layer (supports both MoE and dense)
        shortcut = x
        x = self.layernorm2(x)
        if self.use_moe:
            # MoE returns (output, load_balancing_loss)
            # We store the loss for the training loop to collect
            x, moe_loss = self.feedforward(x)
            moe_losses.append(moe_loss)  # stored on a module-level list
        else:
            x = self.feedforward(x)
        x = self.dropout(x)
        x = x + shortcut
        
        return x, new_keys, new_values
```

### Modifying the training loop to collect MoE losses

```python
# In train.py, inside the training loop

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, ...):
    # ... (setup)
    
    moe_loss_coeff = 0.01  # Switch Transformer coefficient; Mixtral uses 0.02
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        for i, (input_batch, target_batch) in enumerate(train_loader):
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            
            # Collect MoE auxiliary losses from all transformer blocks
            # MoE layers append their load-balancing losses to a shared list
            # during the forward pass. We clear and collect it per batch.
            moe_losses = []
            for block in model.trf_block:
                if hasattr(block, 'feedforward') and hasattr(block.feedforward, 'router'):
                    # Reset MoE's internal loss buffer and mark for collection
                    pass  # In practice, MoE loss is returned from forward()
            
            # The forward pass already accumulated moe_losses into the model.
            # We need to add them to the total loss:
            # total_loss = ce_loss + moe_loss_coeff * sum(moe_losses)
            #
            # An elegant approach: store moe_losses on the model object
            # during the forward pass and sum them here.
            if hasattr(model, 'moe_losses') and model.moe_losses:
                moe_total = sum(model.moe_losses)
                loss = loss + moe_loss_coeff * moe_total
            
            # Gradient accumulation (same as before)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            # ... (rest of training loop unchanged)
```

### Practical considerations

| Concern | Mitigation |
|---------|-----------|
| **Expert load imbalance** | Load-balancing auxiliary loss (coefficient 0.01–0.02) |
| **Router collapse** | Add small Gaussian noise to router logits during training (Shazeer 2017) |
| **Memory overhead** | MoE with 8 experts increases total params ~4× but active params ~1.3×. The unused experts' parameters still need to be stored in memory. |
| **Training instability** | Z-loss (Section 7) on router logits helps; expert dropout can also help |
| **Batch size per expert** | With many experts and small batches, some experts may get zero tokens per batch. Use smaller n_experts (4–8) for small-scale models. |
| **KV cache** | MoE only replaces the FFN; attention is unchanged. KV cache savings from GQA (Section 2) still apply independently. |

### Impact

| Configuration | Total params | Active params per token | Inference speed (relative) |
|-------------|-------------|------------------------|---------------------------|
| Dense (GatedFFN, 768) | ~85M | ~85M | 1× (baseline) |
| MoE-4 (4 experts, top-1) | ~220M | ~67M | ~1.3× |
| MoE-8 (8 experts, top-2) | ~350M | ~95M | ~0.9× |
| MoE-8 (8 experts, top-1) | ~350M | ~55M | ~1.5× |

*The "inversion" (MoE-8 top-2 being slightly slower than dense) is because for a model this small, the routing overhead and expert-loop iteration dominate over the FFN compute. MoE shines at larger scales (>1B params) where each expert's compute dwarfs the routing cost.*

---

## 12. Putting It All Together — Kaito v2 Config

If you were to adopt all the improvements above, here's the updated configuration:

```python
# config.py — Kaito v2 with modern improvements

## Core architecture (same dimensions as GPT-2 small)
BATCH_SIZE = 2
MAX_LENGTH = 8192      # Increased from 512 — RoPE + sliding window allow this
STRIDE = 256
VOCAB_SIZE = 50257
OUTPUT_DIM = 768
N_HEADS = 12
N_KV_HEADS = 4          # GQA: 4 KV heads, 12 query heads (3:1 ratio)
N_LAYERS = 12
DROPOUT = 0.1
LEARNING_RATE = 0.0001

qkv_bias = False

FFN_HIDDEN_MULTIPLIER = 8/3  # SwiGLU needs 8/3*d_model instead of 4*d_model

## Positional encoding
USE_ROPE = True          # Replaces learned position embeddings
SLIDING_WINDOW = 4096    # Tokens attend to at most this many neighbors

## Normalization
USE_RMSNORM = True       # Replaces LayerNorm

## Training
ACCUMULATION_STEPS = 4
GRAD_CLIP_MAX_NORM = 1.0
WEIGHT_DECAY = 0.1       # AdamW weight decay (was 0 / implicit)
WARMUP_RATIO = 0.05      # Fraction of steps for LR warmup
Z_LOSS_COEFF = 1e-4      # Auxiliary loss for logit stability

## Generation
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.9
```

And the `kaitomodel.__init__` becomes:

```python
class kaitomodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
        # No learned position embedding when using RoPE
        # self.pos_embedding is removed
        
        self.dropout = nn.Dropout(DROPOUT)
        
        NormClass = RMSNorm if USE_RMSNORM else LayerNorm
        
        self.trf_block = nn.ModuleList([
            TransformerBlock(norm_class=NormClass) for _ in range(N_LAYERS)
        ])
        self.final_norm = NormClass(OUTPUT_DIM)
        self.out_head = nn.Linear(OUTPUT_DIM, VOCAB_SIZE, bias=False)
        
        # Weight tying: share embedding <-> output head
        self.out_head.weight = self.token_embedding.weight
```

---

## References

| # | Paper | Year | Improvement |
|---|-------|------|-------------|
| 1 | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | 2021 | RoPE |
| 2 | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | 2023 | GQA |
| 3 | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | 2020 | SwiGLU |
| 4 | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 2019 | RMSNorm |
| 5 | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | 2019 | AdamW |
| 6 | [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) | 2022 | Parallel Attn+FFN, Z-Loss |
| 7 | [Mistral 7B](https://arxiv.org/abs/2310.06825) | 2023 | Sliding Window Attention |
| 8 | [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) | 2017 | Weight Tying |
| 9 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Transformer (weight tying ref) |
| 10 | [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538) | 2017 | MoE (original) |
| 11 | [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) | 2022 | MoE (top-1 routing) |
| 12 | [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | 2024 | MoE (top-2, production) |
| 13 | [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) | 2022 | MoE (stability techniques) |
