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
    
    def forward(self, x):
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

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # top left upper triangle

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)   # softmax(attn_scores / sqrt(head_dim))
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        # The output is a new (2, 512, 768) tensor where each token's vector is now context-aware, 
        # containing information from itself and all previous tokens.

        return context_vec