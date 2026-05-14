# flow of this preprocessing text is :
# 1. load text from the file
# 2. clean text using regex, remove all the symbols and keep only the words
# 3. prepare dataloader for the input which we will pass to model for training
#         a. create dataset
#         b. create dataloader 
# 4. embed the input_ids using embedding layer and add position embeddings to it (this is done in the model)

import re
import tiktoken

import torch
from torch.utils.data import Dataset, DataLoader

from config import *
tokenizer = tiktoken.get_encoding("gpt2")

# define how your data is stored and how it's accessed
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        print(f"[DEBUG] Tokenized text length: {len(token_ids)}")

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        print(f"[DEBUG] size of token_ids: {len(token_ids) - MAX_LENGTH}")
        for i in range(0, len(token_ids) - MAX_LENGTH, STRIDE):
            input_chunk = token_ids[i:i + MAX_LENGTH]
            target_chunk = token_ids[i + 1: i + MAX_LENGTH + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        # input = [0,1,2,3]
        # target = [1,2,3,4]
        print(f"[DEBUG] Number of chunks: {len(self.input_ids)}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class PreprocessText:
    def __init__(self, file_path = "data_prep/the-verdict.txt"):
        self.file_path = file_path
        self.text = ""
    
    def load_text(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            self.text = file.read()
        
    def _split_and_join(self, text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        result = [item.strip() for item in result if item.strip()]
        return " ".join(result)

    def clean_text(self):
        self.text = self._split_and_join(self.text)

    def tokenize_text(self, text):
        cleaned = self._split_and_join(text)
        return tokenizer.encode(cleaned, allowed_special={"<|endoftext|>"})
    
    # Provides batching, shuffling, and parallel loading of data from a Dataset
    def create_dataloader(self, shuffle=True, drop_last=True, num_workers=0, train_test_split = 0.9):
        print(f"[DEBUG] Number of tokens in text: {len(tokenizer.encode(self.text))}")

        # Create dataset
        dataset = GPTDataset(self.text, tokenizer)
        total_samples = len(dataset)
        train_size = int(train_test_split * total_samples)
        test_size = total_samples - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # training dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        # testing dataloader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return train_dataloader, test_dataloader
    
    def preprocess(self, text=None):
        if text is not None:
            token_ids = self.tokenize_text(text)
            return torch.tensor(token_ids).unsqueeze(0)
        else:
            self.load_text()
            self.clean_text()
            return self.create_dataloader()

### sinusoidal positional embedding:-

# PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

# where pos is the position, i is the index of the dimension vector embedding, and d_model is the output dimension of the embedding layer.
# 10000^2i/d​ ensures that each dimension of the positional embedding has a different frequency.
# why use this ? this gives unqiue and relative position to each word in the sequence.

### rotary positional embedding:-
# does not add positional information to the embeddings. 
# Instead, it rotates the Query and Key vectors based on their absolute position.

# its like rotating the vector(quesry(m) and key(n)) in 2D space based on its position with theta angle.
# when we do dot product of query and key, it is the function of the m-n
# [https://arxiv.org/pdf/2104.09864]

### learned positional embedding

# initialize the positional embedding matrix with random values
# learn the positional embedding matrix during training

# final_embedding = word_embedding + positional_embedding{learned, embedding}

### relative positional embedding:-
#  the attention score between two tokens should not depend on their absolute positions, 
# but rather on the offset or distance between them.

# Instead of adding positional information to the initial word embeddings, RPE injects the positional information directly into the attention score calculation.

# The standard attention score between a query vector Qi (at position i) and a key vector Kj (at position j) is calculated by their dot product: 
# Scoreij = Qi ⋅ Kj

# With RPE, we introduce a bias term that is based on the relative distance j-i. 
# The model learns a unique embedding vector for each possible relative position
# New_score = Qi ⋅ Kj + b(j-i) {trainable parameter like penalty added to attention score}
# It directly tells the attention mechanism to "add a bonus/penalty to your score based on how far apart these two words are.