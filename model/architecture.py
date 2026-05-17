### Here i want to build the model architecture
# 1. layernorm / rmsnorm 
# why use rmsnorm? -> Normalize each sample individually, across the feature dimension.(row level) (best for rnn,transformer)
# RMSNorm is a simpler alternative to LayerNorm — it removes the mean-centering step
# and only normalises by the root-mean-square. This is ~15-25% faster and empirically
# performs just as well (used by Llama, Mistral, Gemma).
# where as in batchnorm -> we normalize each feature, across the batch dimension.(column level) (best for cnn)

# 2. feedforward (gated: SwiGLU)
# 3. skip connection
# 4. transformer block 

from config import *
from multi_attention.attention import grouped_query_attention
from model.moe import MoELayer
from model.ffn import GatedFeedForward
import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not subtract the mean:
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * scale

    Why does this work without centering?
        In deep transformers, the pre-norm residual connections ensure the
        input to each sub-layer is already roughly zero-mean (residual streams
        average over many contributions). The remaining variance mismatch is
        what needs normalising, and RMSNorm handles that without computing
        the mean — saving ~15-25% of the normalisation FLOPs.
    
    The scale parameter is still learnable because the optimal variance
    for attention softmax (scale-sensitive) differs from the FFN (activation
    function-based). Llama, Mistral, and Gemma all use RMSNorm.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5
        # Learnable scale parameter (no shift — RMSNorm doesn't center).
        # scale allows the model to learn the optimal standard deviation
        # for the activations after normalisation.
        self.scale = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        # rms = sqrt(mean(x^2) + eps)
        # We add eps inside the sqrt (matching LayerNorm convention)
        # to avoid division by zero on zero-valued inputs.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


class TransformerBlock(nn.Module):
    def __init__(self, use_moe=False):
        super().__init__()
        # Replaced multihead_attention with grouped_query_attention (GQA + RoPE).
        # GQA reduces KV cache size by using fewer key/value heads than query heads.
        self.attention = grouped_query_attention(
            d_input = OUTPUT_DIM,
            d_output = OUTPUT_DIM,
            context_length = MAX_LENGTH,
            num_heads = N_HEADS,
            n_kv_heads = N_KV_HEADS,  # GQA: fewer KV heads for smaller cache
            dropout = DROPOUT,
            qkv_bias = qkv_bias, # False
            sliding_window = SLIDING_WINDOW  # None = full causal; int = windowed
        )
        # Replaced LayerNorm with RMSNorm (faster, empirically equivalent).
        # Pre-normalisation before attention and FFN (standard for modern transformers).
        self.layernorm1 = RMSNorm(OUTPUT_DIM)
        self.layernorm2 = RMSNorm(OUTPUT_DIM)
        # MoE replaces the dense FFN with a sparse set of expert FFNs.
        # When USE_MOE is True, each token is routed to ~top_k experts,
        # enabling a larger total parameter count at constant compute.
        if use_moe:
            self.feedforward = MoELayer()
        else:
            # Replaced standard FeedForward (GELU) with GatedFeedForward (SwiGLU).
            # Same parameter count, better quality per parameter.
            self.feedforward = GatedFeedForward()
        # Track whether this block uses MoE (needed by training loop for aux loss).
        self.use_moe = use_moe
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, past_keys=None, past_values=None):
        """
        x: input tensor (batch, num_tokens, embed_dim)
        past_keys / past_values: KV cache from previous generation steps (or None during training).
        Returns: output tensor, updated keys, updated values (for KV-cache chain).
        """
        # Parallel formulation (PaLM-style; Chowdhery et al., 2022).
        # Attention and FFN share the same normalised input and contribute
        # independently to the residual stream in one step:
        #   x = x + attention(norm(x)) + ffn(norm(x))
        #
        # This removes one sequential dependency per transformer block —
        # instead of two serial kernel launches, the GPU can overlap
        # attention and FFN computation, yielding ~15% faster training
        # without measurable quality loss.
        #
        # Note: layernorm2 is kept as a parameter for checkpoint compatibility
        # but is unused in this forward path. It will not receive gradients.
        normed = self.layernorm1(x)
        attn_out, new_keys, new_values = self.attention(normed, past_keys, past_values)
        attn_out = self.dropout(attn_out)
        ffn_out = self.dropout(self.feedforward(normed))
        x = x + attn_out + ffn_out
        return x, new_keys, new_values
