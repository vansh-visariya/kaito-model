# Data & Training Guide for Kaito Model

## Recommended Datasets

The Kaito model (124M params, GPT-2 small architecture) works well with any text dataset
suitable for autoregressive language modelling. Below are freely available datasets ordered
by practicality for this model size.

### Quick comparison

| Dataset | Size | Tokens | License | Download effort |
|---|---|---|---|---|
| [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) | 38 GB | ~9B | CC0 | Medium |
| [The Pile](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) | 825 GB | ~300B | MIT | High (use subset) |
| [FineWeb (sample-10B)](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | ~40 GB | 10B | ODC-By | Medium |
| [WikiText-103](https://huggingface.co/datasets/wikitext) | 516 MB | ~103M | CC BY-SA 4.0 | Trivial |
| [C4 (en)](https://huggingface.co/datasets/allenai/c4) | 750 GB | ~200B | ODC-By | High (use sample) |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 2.7 GB | ~500M | Apache 2.0 | Easy |
| [RedPajama (sample)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | 100B+ | 1T+ | Apache 2.0 | Very high |
| [Project Gutenberg](https://www.gutenberg.org/) | ~60 GB | ~10B | Public domain | Manual |

### Best choices for Kaito model

1. **OpenWebText** — The closest match to GPT-2's original training distribution.
   ~9B tokens is ideal for a single-node training run of a 124M model (roughly
   70× data-to-parameter ratio, matching Chinchilla scaling).

2. **FineWeb sample-10B** — Higher quality than OpenWebText (better dedup, better
   filtering). Same token budget, same training cost. The 10B-token sample
   (`sample-10B`) is exactly the right size.

3. **WikiText-103** — Great for quick validation runs. Small enough to download
   in seconds, but too small for serious training (~103M tokens, <1× data-to-params).

4. **TinyStories** — Synthetic stories designed for small LMs. Good for debugging
   training pipelines. Limited to simple vocabulary and short contexts.

---

## Data Pipeline

### Current pipeline (single file)

The existing `PreprocessText` in `data_prep/preprocess_text.py`:

1. Loads a .txt file
2. Tokenises it with GPT-2 tokenizer (tiktoken)
3. Chunks into overlapping sequences via sliding window
4. Splits into train/validation DataLoaders

**Usage:**
```python
from data_prep.preprocess_text import PreprocessText
preprocessor = PreprocessText("path/to/your/file.txt")
train_loader, val_loader = preprocessor.preprocess()
```

### Pipeline for large datasets (recommended)

For datasets larger than a single file, use HuggingFace `datasets`:

```python
import torch
from torch.utils.data import DataLoader, Dataset
from config import MAX_LENGTH, STRIDE, BATCH_SIZE
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

class StreamingTextDataset(Dataset):
    """
    Memory-efficient dataset that tokenises and chunks on-the-fly.
    Works with HuggingFace datasets or any iterable of texts.
    """
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH, stride=STRIDE):
        self.input_ids = []
        self.target_ids = []
        for text in texts:
            token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            for i in range(0, len(token_ids) - max_length, stride):
                self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
                self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloaders_from_hf(dataset_name="HuggingFaceFW/fineweb",
                                split="train", sample_size=100_000,
                                train_split=0.9):
    """
    Load a HuggingFace dataset, tokenise, and create DataLoaders.
    Adjust `sample_size` based on your available RAM/VRAM.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split, streaming=True)
    texts = []
    for i, example in enumerate(ds):
        if i >= sample_size:
            break
        texts.append(example["text"])

    full_dataset = StreamingTextDataset(texts, tokenizer)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return train_loader, val_loader
```

### Using your own text files

Place any number of .txt files in a directory and load them:

```python
import os

def load_texts_from_dir(directory):
    texts = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".txt"):
            with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

texts = load_texts_from_dir("data_prep/my_corpus/")
dataset = StreamingTextDataset(texts, tokenizer)
# ... then DataLoader as above
```

---

## Training Commands

### Basic training (the-verdict.txt, current default)

```bash
python train.py
```

### Training on a different dataset

There are two approaches:

**A) Modify `config.py` training hyperparameters:**
```python
BATCH_SIZE = 8              # larger batch if you have VRAM
MAX_LENGTH = 512
ACCUMULATION_STEPS = 4      # effective batch = BATCH_SIZE * ACCUMULATION_STEPS
LEARNING_RATE = 3e-4        # standard LR for GPT-2 scale models
```

**B) Write a standalone training script:**
```python
import torch
from config import *
from main import kaitomodel
from data_prep.preprocess_text import PreprocessText
from train import train_model
import tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

# Option 1: single file (download any .txt corpus)
preprocessor = PreprocessText("path/to/large-corpus.txt")
train_loader, val_loader = preprocessor.preprocess()

# Option 2: HuggingFace dataset (install: pip install datasets)
# from create_dataloaders_from_hf import create_dataloaders_from_hf
# train_loader, val_loader = create_dataloaders_from_hf("Skylion007/openwebtext")

model = kaitomodel()
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params:,} parameters")

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=1,
    eval_freq=200,
    eval_iter=50,
    start_context="The meaning of life is",
    tokenizer=tokenizer,
)
```

---

## Dataset Download Instructions

### OpenWebText (recommended)

```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
for i, example in enumerate(ds):
    if i >= 1000: break
    # texts are in example['text']
"
```

### FineWeb sample-10B

```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb', 'sample-10B', split='train', streaming=True)
for i, example in enumerate(ds):
    if i >= 1000: break
    print(example['text'][:200])
"
```

### WikiText-103

```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
print(len(ds), 'articles')
"
```

### The Pile (use a subset)

```bash
pip install datasets
python -c "
from datasets import load_dataset
# Only load 50k examples for a quick run
ds = load_dataset('monology/pile-uncopyrighted', split='train', streaming=True)
texts = [next(ds)['text'] for _ in range(50000)]
"
```

---

## Hardware Requirements

| Model variant | Params | VRAM (batch=2, len=512) | VRAM (batch=8, len=1024) | Recommended GPU |
|---|---|---|---|---|
| Dense (USE_MOE=False) | 114M | ~3.5 GB | ~14 GB | RTX 3060+ |
| MoE (USE_MOE=True) | 510M | ~8 GB | ~28 GB | RTX 3090+ |

- Use `BATCH_SIZE=1` and gradient accumulation for larger models on limited VRAM.
- Enable mixed precision (bfloat16) automatically on Ampere+ GPUs.

---

## Expected Training Time

Estimated wall-clock time for 1 epoch on **OpenWebText (~9B tokens)** with various GPUs:

| GPU | Dense (114M) | MoE (510M) |
|---|---|---|
| RTX 3090 (24 GB) | ~3 days | ~7 days |
| A100 (80 GB) | ~12 hours | ~28 hours |
| RTX 4090 (24 GB) | ~2 days | ~5 days |

For a quick sanity check (loss should drop from ~11 to ~4-5), train on 1M tokens first.

---

## Data Preprocessing Notes

- **Tokenisation**: The model uses `tiktoken` GPT-2 tokenizer (50,257 vocab). Always use
  `tiktoken.get_encoding("gpt2")` for consistency.
- **Sliding window**: `MAX_LENGTH=512` with `STRIDE=256` chunks each document into
  overlapping sequences. Documents shorter than `MAX_LENGTH` are skipped.
- **Document separator**: Append `<|endoftext|>` between documents in multi-document
  corpora so the model learns the document boundary token.
- **Train/validation split**: Currently 90/10 random split. For large datasets, use
  99/1 or hold out a fixed validation set.
- **Mixed precision**: bfloat16 autocast is enabled automatically on CUDA — no action needed.
