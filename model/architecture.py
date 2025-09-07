### Here i want to build the model architecture
# 1. layernorm 
# why use layernorm? -> Normalize each sample individually, across the feature dimension.(row level) (best for rnn,transformer)
# where as in batchnorm -> we normalize each feature, across the batch dimension.(column level) (best for cnn)

# 2. feedforward 
# 3. skip connection
# 4. transformer block 

from config import *
import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super.__init__()
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
        var = x.var(dim=-1, keepdim=True)
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
        super.__init__()
    
    def forward(self, x):
        # gelu approximation
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self):
        super.__init__()
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
        super.__init__()
    
    def forward(self, x):
        pass
