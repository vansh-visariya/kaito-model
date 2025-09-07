from config import *
import torch
import tiktoken

encoding = tiktoken.get_encoding("gpt2")

def generate_text(model,embedder, prompt, new_max_length=200):
    for _ in range(new_max_length):
        with torch.no_grad():
            input_tensor = embedder.preprocess(text = prompt)
            logit = model(input_tensor)               # output shape: [batch, seq_len, vocab_size]
            logit = logit[:, -1, :]            # last token logits: [batch, vocab_size]
            probs = torch.softmax(logit, dim=-1)
            next_input = torch.argmax(probs, dim=-1, keepdim=True)  # [batch, 1]
            word = next_input.tolist()
            next_word = encoding.decode(word[-1])
            print(f"next word:{next_word}")
            prompt = prompt + next_word  # extend the prompt
    return prompt

from data_prep.preprocess_text import PreprocessText
embedder = PreprocessText()

from main import kaitomodel
model = kaitomodel()

out = generate_text(model, embedder, prompt = "hello, goood", new_max_length=10)
print(out)