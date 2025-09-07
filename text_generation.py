from config import *
import torch
import tiktoken

encoding = tiktoken.get_encoding("gpt2")

def generate_text(model, embedder, prompt, new_max_length=200):
    for _ in range(new_max_length):
        with torch.no_grad():
            input_ids = embedder.preprocess(text=prompt)  # Now returns token IDs
            logits = model(input_ids)  # Model handles embedding internally
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            next_word = encoding.decode([next_token.item()])
            prompt = prompt + next_word
    return prompt

from data_prep.preprocess_text import PreprocessText
embedder = PreprocessText()

from main import kaitomodel
model = kaitomodel()

out = generate_text(model, embedder, prompt = "hello, goood", new_max_length=10)
print(out)
