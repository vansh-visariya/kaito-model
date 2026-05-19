from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset, DataLoader
import tiktoken
from config import *

class FineWebStreamingDataset(IterableDataset):
    def __init__(self, split="train", tokenizer=None, max_length=MAX_LENGTH, stride=MAX_LENGTH):
        """
        Streaming dataset for HuggingFaceFW/fineweb (sample-10BT).
        Uses non-overlapping chunks by default (stride = max_length).
        """
        super().__init__()
        # Use a specific configuration of FineWeb
        self.dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split=split, streaming=True)
        self.tokenizer = tokenizer if tokenizer else tiktoken.get_encoding("gpt2")
        self.max_length = max_length
        self.stride = stride

    def __iter__(self):
        buffer = []
        # Get the token ID for <|endoftext|>
        eot_token = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        for item in self.dataset:
            # Tokenize the current document
            tokens = self.tokenizer.encode(item['text'])
            tokens.append(eot_token) # Separate documents with EOT
            buffer.extend(tokens)
            
            # Yield chunks of size (MAX_LENGTH + 1) for input and target
            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                # Slide the buffer by STRIDE (using non-overlapping chunks, so stride=max_length)
                buffer = buffer[self.stride:]
                
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, target_ids

def create_fineweb_dataloaders(tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH, stride=MAX_LENGTH):
    """
    Creates streaming dataloaders for training and validation.
    Note: FineWeb doesn't have a default validation split in the 10BT sample, 
    so we take a small portion for validation by skipping the train data.
    """
    # Create the training dataset stream
    train_dataset = FineWebStreamingDataset(split="train", tokenizer=tokenizer, max_length=max_length, stride=stride)
    
    # For validation, we could theoretically use a different dataset or just 
    # skip a chunk of fineweb to create a disjoint set.
    val_dataset = FineWebStreamingDataset(split="train", tokenizer=tokenizer, max_length=max_length, stride=stride)
    
    # We skip the first 100,000 items in the val stream to ensure it's disjoint from the start of the train stream.
    # (In a real massive training run, you'd use a dedicated eval dataset)
    val_dataset.dataset = val_dataset.dataset.skip(100000)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
