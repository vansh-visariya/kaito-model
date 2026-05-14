from config import *
import torch

def generate_text(model, prompt, tokenizer, new_max_length=200):
    model.eval()
    token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    for _ in range(new_max_length):
        with torch.no_grad():
            if token_ids.size(1) > MAX_LENGTH:
                token_ids = token_ids[:, -MAX_LENGTH:]
            logits = model(token_ids)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            token_ids = torch.cat([token_ids, next_token], dim=1)
    model.train()
    return tokenizer.decode(token_ids.squeeze(0).tolist())

# from data_prep.preprocess_text import PreprocessText
# embedder = PreprocessText()

# from main import kaitomodel
# model = kaitomodel()

# out = generate_text(model, embedder, prompt = "hello, goood", new_max_length=10)
# print(out)