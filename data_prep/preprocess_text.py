# flow of this preprocessing text is :
# 1. load text from the file
# 2. clean text using regex, remove all the symbols and keep only the words
# 3. prepare dataloader for the input which we will pass to model for training
#         a. create dataset
#         b. create dataloader 
# 4. embed the input_ids using embedding layer and add position embeddings to it

import re
import tiktoken

import torch
from torch.utils.data import Dataset, DataLoader

from config import *

# define how your data is stored and how it's accessed
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - MAX_LENGTH, STRIDE):
            input_chunk = token_ids[i:i + MAX_LENGTH]
            target_chunk = token_ids[i + 1: i + MAX_LENGTH + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        # input = [0,1,2,3]
        # target = [1,2,3,4]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class PreprocessText:
    def __init__(self, file_path = "./the-verdict.txt"):
        self.file_path = file_path
        self.text = ""
    
    def load_text(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            self.text = file.read()
        
    def clean_text(self):
        # Split the text into words, keeping the punctuation and whitespace
        result = re.split(r'([,.:;?_!"()\']|--|\s)', self.text)
        result = [item.strip() for item in result if item.strip()]
        self.text = " ".join(result)
    
    # Provides batching, shuffling, and parallel loading of data from a Dataset
    def create_dataloader(self, shuffle=True, drop_last=True, num_workers=0):
        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset
        dataset = GPTDataset(self.text, tokenizer)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader
    
    def embedding(self, dataloader):
        embedding_layer = torch.nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)
        all_embedded_inputs = []

        # Iterate over the dataloader and embed the input_ids
        for input_ids, _ in dataloader:
            embedded_input = embedding_layer(input_ids)
            all_embedded_inputs.append(embedded_input)
        
        ## Here i can also use different type of positional embedding like sinusoidal embedding and rotary embedding
        # This is learned positional embedding
        pos_embedding_layer = torch.nn.Embedding(MAX_LENGTH, OUTPUT_DIM)

        # Get the position embeddings
        pos_embeddings = pos_embedding_layer(torch.arange(MAX_LENGTH))
        # Add the position embeddings to all the input embeddings
        input_embeddings = all_embedded_inputs + pos_embeddings

        # Concatenate all the embedded inputs and dimension keeps to 0 so that it doesn't change the batch size
        # return size (num_batches × batch_size, max_length, output_dim)
        return input_embeddings