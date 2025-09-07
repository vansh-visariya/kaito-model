import torch
import torch.nn as nn
from config import *
from model.architecture import TransformerBlock, LayerNorm
from data_prep.preprocess_text import PreprocessText

class kaitomodel(nn.Module):
    def __init__(self):
        super().__init__()
        # Add embedding layers to the model
        self.token_embedding = nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
        self.pos_embedding = nn.Embedding(MAX_LENGTH, OUTPUT_DIM)
        
        self.dropout = nn.Dropout(DROPOUT)
        self.trf_block = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])
        self.final_norm = LayerNorm(OUTPUT_DIM)
        self.out_head = nn.Linear(OUTPUT_DIM, VOCAB_SIZE, bias=False)
    
    def forward(self, input_ids):
        # input_ids shape: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Get position embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_embeds = self.pos_embedding(positions)  # [seq_len, embed_dim]
        
        # Combine embeddings
        x = token_embeds + pos_embeds  # Broadcasting: [batch_size, seq_len, embed_dim]
        
        x = self.dropout(x)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
