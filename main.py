import torch
import torch.nn as nn
from config import *
from model.architecture import TransformerBlock, LayerNorm
from data_prep.preprocess_text import PreprocessText

class kaitomodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT)
        self.trf_block = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])
        self.final_norm = LayerNorm(OUTPUT_DIM)
        self.out_head = nn.Linear(OUTPUT_DIM, VOCAB_SIZE, bias=False)
    
    def forward(self, input_ids):
        x = self.dropout(input_ids)
        x = self.trf_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)  # logits = [num_batches × batch_size, max_length, vocab_size]
        return logits
