import torch
import torch.nn as nn
from config import *
from model.architecture import TransformerBlock
from data_prep.preprocess_text import PreprocessText

class kaitomodel:
    def __init__(self):
        self.dropout = nn.Dropout(DROPOUT)

        self.trf_block = nn.Sequential()