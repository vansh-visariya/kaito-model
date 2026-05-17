### Mixture of Experts (MoE) layer for sparse FFN computation.
#
# Instead of running every input through a single dense FFN, MoE routes
# each token to a small subset of expert FFNs (top-k routing). This allows
# the total parameter count to grow while keeping inference compute flat.
#
# References:
#   - Shazeer et al., 2017: "Outrageously Large Neural Networks"
#   - Mistral-8x7B (Jiang et al., 2024): top-2 routing with load balancing
#   - DeepSeek-V2 (2024): fine-grained expert routing
#
# The load-balancing loss follows the "auxiliary loss" approach from
# the original MoE paper. For each token, we compute the fraction of
# router probability assigned to each expert and the fraction of tokens
# actually routed to each expert. The auxiliary loss is:
#   N_EXPERTS * sum_i(f_i * p_i)
# where f_i = fraction of tokens sent to expert i, and
#       p_i = average router probability for expert i.
# Minimising this encourages uniform expert usage.

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_EXPERTS, MOE_TOP_K, OUTPUT_DIM
from model.ffn import GatedFeedForward


class MoELayer(nn.Module):
    """
    Sparse Mixture of Experts that replaces a standard FFN.

    Each token is independently routed to the top-k experts (by router
    probability), and the expert outputs are weighted by their gating
    probabilities. The router is a learned linear projection from d_model
    to n_experts followed by softmax.

    The auxiliary load-balancing loss is stored in `self.last_aux_loss`
    after each forward pass, so the training loop can sum and add it to
    the total loss.
    """

    def __init__(self, d_model: int = OUTPUT_DIM):
        super().__init__()
        # Router: learns a probability distribution over experts per token.
        # No bias so that router logits are purely a learned similarity
        # between the token embedding and each expert "direction".
        self.router = nn.Linear(d_model, N_EXPERTS, bias=False)
        # Initialise router weights with small variance to avoid extreme
        # logits at initialisation (which make softmax near-one-hot and
        # kill exploration of non-top-1 experts).
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        # Expert FFNs: each is a full SwiGLU GatedFeedForward.
        # Using ModuleList rather than a single nn.ModuleList because we
        # need to loop over experts by index for top-k dispatch.
        self.experts = nn.ModuleList(
            [GatedFeedForward() for _ in range(N_EXPERTS)]
        )

        # Placeholder for the load-balancing auxiliary loss; updated
        # every forward pass so the training loop can read it after
        # the model forward.
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)

        Returns: (batch, seq_len, d_model) — output from the routed experts.
        """
        batch, seq_len, d_model = x.shape
        num_tokens = batch * seq_len

        # Flatten to (num_tokens, d_model) for per-token routing.
        x_flat = x.view(-1, d_model)

        # Router logits: (num_tokens, n_experts)
        router_logits = self.router(x_flat)

        # Softmax over experts to get routing probabilities.
        router_probs = F.softmax(router_logits, dim=-1)  # (num_tokens, n_experts)

        # --- Load-balancing auxiliary loss ---
        # f_i: fraction of tokens routed to expert i (based on top-1 selection).
        # p_i: average router probability assigned to expert i across all tokens.
        #
        # The loss N_EXPERTS * sum(f_i * p_i) is minimised when routing is
        # uniform (f_i = p_i = 1/N_EXPERTS, so sum = 1/N_EXPERTS, loss = 1).
        # It is maximised when routing is concentrated on one expert.
        #
        # f_i depends on the discrete top-1 choice (not differentiable), so we
        # detach it. p_i is a differentiable function of the router weights via
        # softmax, so gradients flow through p_i to update the router.
        with torch.no_grad():
            # Which expert does each token select as its top-1?
            # (We use the top-1 for the load-balancing loss, following the
            # convention from the original MoE paper and Mistral-8x7B.)
            _, top1_idx = router_probs.topk(1, dim=-1)  # (num_tokens, 1)
            top1_idx = top1_idx.squeeze(-1)              # (num_tokens,)

            # f_i: fraction of tokens that picked expert i (no grad — discrete op).
            f_i = torch.zeros(N_EXPERTS, device=x.device)
            f_i.scatter_add_(0, top1_idx,
                             torch.ones(num_tokens, device=x.device) / num_tokens)

        # p_i: average router probability — keep gradients flowing through router.
        p_i = router_probs.mean(dim=0)  # (n_experts,)

        # Auxiliary loss: encourage balanced routing.
        # Multiplication by N_EXPERTS normalises the loss: if all experts
        # are equally used, aux_loss = N_EXPERTS * (1/N_EXPERTS) = 1.
        self.last_aux_loss = N_EXPERTS * (f_i * p_i).sum()
        # --- end load balancing ---

        # --- Top-k expert selection ---
        # Weights and indices of the top-k experts for each token.
        top_k_probs, top_k_indices = router_probs.topk(MOE_TOP_K, dim=-1)
        # Normalise: the selected experts' probabilities are re-normalised
        # to sum to 1 (since we only use top-k, the remaining mass is dropped).
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        # --- end top-k ---

        # --- Sparse expert computation ---
        # Initialise output buffer (same shape as input).
        output = torch.zeros_like(x_flat)  # (num_tokens, d_model)

        # For each expert, find which tokens use it and accumulate its
        # contribution. This loop is the standard "expert dispatch" and is
        # O(N_EXPERTS * num_tokens_per_expert * d_model) — the dominant
        # cost of the MoE layer.
        for expert_idx in range(N_EXPERTS):
            # Mask of tokens that have this expert in their top-k.
            mask = (top_k_indices == expert_idx)  # (num_tokens, top_k)
            if not mask.any():
                continue

            # Which (token, top-k-slot) pairs map to this expert?
            # mask has shape (num_tokens, top_k). We need token indices
            # for each positive entry. We use nonzero to get the 2D indices.
            token_positions, _ = mask.nonzero(as_tuple=True)  # indices along dim 0

            # Gather the gating weights for these tokens from this expert.
            gate_weights = top_k_probs[token_positions, _]  # (num_selected_tokens,)

            # Run the expert and weight its output.
            expert_out = self.experts[expert_idx](x_flat[token_positions])  # (num_selected_tokens, d_model)
            output.index_add_(0, token_positions, gate_weights.unsqueeze(-1) * expert_out)
        # --- end sparse computation ---

        # Restore batch/seq_len shape.
        return output.view(batch, seq_len, d_model)
