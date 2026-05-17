### Here i want to build the model architecture
# 1. layernorm 
# why use layernorm? -> Normalize each sample individually, across the feature dimension.(row level) (best for rnn,transformer)
# where as in batchnorm -> we normalize each feature, across the batch dimension.(column level) (best for cnn)

# 2. feedforward 
# 3. skip connection
# 4. transformer block 

import math
from config import *
from multi_attention.attention import multihead_attention
import torch.nn as nn
import torch

# Precompute GELU constant so it's allocated once, not every forward pass.
# Using math.sqrt avoids creating a fresh torch.tensor each time (performance ~10ns per call).
# Additionally, Python floats automatically move to the correct device when used with torch tensors.
_GELU_CONST = math.sqrt(2.0 / math.pi)

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5
        # scale and shift helps to controling covariate shift
        # covariate shift is a change in the distribution of the input data to a model, but same output function
        # which can cause the model to perform poorly.

        # scale allows the model to learn the optimal standard deviation for the activations.
        self.scale = nn.Parameter(torch.ones(embedding_dim))

        # shift allows the model to learn the optimal mean for the activations.
        self.shift = nn.Parameter(torch.zeros(embedding_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False matches nn.LayerNorm behaviour (biased variance).
        # PyTorch's var defaults to unbiased=True (Bessel's correction),
        # which gives a slightly different normalisation and breaks
        # compatibility with pretrained weights. Using biased variance
        # is the standard convention in Transformer implementations.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normed = (x - mean) / torch.sqrt(var + self.eps)
        # scale and shift help to undo the normalization, 
        # bcs sometimes due to normalization the model might learn to ignore some features.
        return self.scale * x_normed + self.shift

# we use gelu instead of relu,
# gelu is more smooth and continuous than relu.
# for a small negative, relu will output 0, but gelu will output a small negative value.
# which helps the model to learn better.
class Gelu(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # gelu approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # The constant sqrt(2/pi) is precomputed as _GELU_CONST (module-level float)
        # to avoid creating a new CPU tensor on every forward call.
        return 0.5 * x * (1 + torch.tanh(_GELU_CONST * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # this enlargement and again to original dimension helps the model to learn better.
        # its like trying to extract as much information as possible from the input.
        self.layer = nn.Sequential(
            nn.Linear(OUTPUT_DIM, 4 * OUTPUT_DIM), # 768 -> 3072
            Gelu(),
            nn.Linear(4 * OUTPUT_DIM, OUTPUT_DIM) # 3072 -> 768
        )
    
    def forward(self, x):
        return self.layer(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = multihead_attention(
            d_input = OUTPUT_DIM,
            d_output = OUTPUT_DIM,
            context_length = MAX_LENGTH,
            num_heads = N_HEADS,
            dropout = DROPOUT,
            qkv_bias = qkv_bias # False
        )
        self.layernorm1 = LayerNorm(OUTPUT_DIM)
        self.layernorm2 = LayerNorm(OUTPUT_DIM)
        self.feedforward = FeedForward()
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, past_keys=None, past_values=None):
        """
        x: input tensor (batch, num_tokens, embed_dim)
        past_keys / past_values: KV cache from previous generation steps (or None during training).
        Returns: output tensor, updated keys, updated values (for KV-cache chain).
        """
        # shortcut connection 1
        shortcut = x
        x = self.layernorm1(x)
        x, new_keys, new_values = self.attention(x, past_keys, past_values)
        x = self.dropout(x)
        x = x + shortcut

        # shortcut connection 2
        shortcut = x
        x = self.layernorm2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x, new_keys, new_values
