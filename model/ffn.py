### Feed-forward network components used by both the dense and MoE paths.
#
# SwiGLU activation and GatedFeedForward are defined here to break the
# circular dependency between architecture.py (TransformerBlock) and
# moe.py (MoELayer), both of which need GatedFeedForward.

import torch.nn as nn
import torch
from config import OUTPUT_DIM


# SwiGLU activation: swish(x) = x * sigmoid(x)
# Used in the gated FFN (GatedFeedForward below).
# GELU and Swish are extremely similar (both are smooth, non-monotonic,
# bounded below). Swish is slightly cheaper (sigmoid is simpler than
# GELU's tanh-of-cubic approximation) and SwiGLU (the gated variant)
# was shown by Shazeer (2020) to outperform GELU-gated variants.
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish(x) = x * sigmoid(x)
        return x * torch.sigmoid(x)


class GatedFeedForward(nn.Module):
    """
    SwiGLU gated feed-forward network.

    Standard FFN: output = W2(GELU(W1(x)))
    SwiGLU FFN:   output = W3(Swish(W_gate(x)) * W_up(x))

    The gate and up projections run in parallel; their element-wise product
    forms the gating mechanism. This lets the network learn which information
    to pass through and which to suppress — more expressive than a pointwise
    activation.

    Why a smaller hidden dim (8/3 * d_model instead of 4 * d_model)?
        SwiGLU has three weight matrices (gate, up, down) vs two (in, out)
        in the standard FFN. To match the same total parameter count, the
        hidden dimension is reduced by ~2/3. For d_model=768:
          Standard:  768 * 3072 + 3072 * 768 = 4,718,592 params
          SwiGLU:    768 * 2048 + 768 * 2048 + 2048 * 768 = 4,718,592 params
        Same parameter budget, better quality (Shazeer 2020).
    """
    def __init__(self):
        super().__init__()
        # Llama convention: hidden_dim = int(8/3 * d_model).
        # For 768: 8/3 * 768 = 2048, a GPU-tensor-core-friendly size.
        hidden_dim = int(8/3 * OUTPUT_DIM)

        # Gate projection: produces the "weights" of the gate.
        # Gate output passes through SwiGLU activation.
        self.W_gate = nn.Linear(OUTPUT_DIM, hidden_dim, bias=False)
        # Up projection: produces the "values" to be gated.
        # Gate * Up forms the expressive gating interaction.
        self.W_up = nn.Linear(OUTPUT_DIM, hidden_dim, bias=False)
        # Down projection: maps back from hidden dim to d_model.
        self.W_down = nn.Linear(hidden_dim, OUTPUT_DIM, bias=False)

        self.activation = SwiGLU()

    def forward(self, x):
        # Gate: which features to keep (after SwiGLU non-linearity).
        gate_out = self.activation(self.W_gate(x))
        # Up: candidate feature values.
        up_out = self.W_up(x)
        # Element-wise gating: scale each candidate feature by its
        # "relevance" signal from the gate. This is more expressive
        # than a simple pointwise activation because each feature's
        # scaling factor is independently learned.
        gated = gate_out * up_out
        return self.W_down(gated)
