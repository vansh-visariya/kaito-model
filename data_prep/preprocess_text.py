import re
import tiktoken

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
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
    
    def load_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.text = file.read()
        
    def clean_text(self):
        # Split the text into words, keeping the punctuation and whitespace
        result = re.split(r'([,.:;?_!"()\']|--|\s)', self.text)
        result = [item.strip() for item in result if item.strip()]
        self.text = " ".join(result)
    
    
    def create_dataloader(self, batch_size=4, max_length=256, 
                         stride=128 , shuffle=True, drop_last=True,
                         num_workers=0):

        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset
        dataset = GPTDataset(self.text, tokenizer, max_length, stride)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader