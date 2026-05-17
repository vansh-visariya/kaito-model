# Kaito Model — GPT-2 from Scratch with Modern Improvements

A GPT-2 style decoder-only transformer (124M params) built from scratch in PyTorch, then upgraded with modern LLM techniques: **RoPE, GQA, SwiGLU, RMSNorm, Sliding Window Attention, Parallel Attention+FFN, Mixed Precision, Weight Tying, Z-Loss, and optional MoE**.

[Model on HuggingFace](https://huggingface.co/vansh-myth/kaito)

---

## Features

- **Full GPT-2 scale** — 12 layers, 12 heads, 768-dim, 114M params (510M with MoE)
- **RoPE** (Rotary Position Embedding) — length extrapolation, no learned positions
- **GQA** (Grouped Query Attention) — 12 query heads share 4 KV heads; 3× smaller KV cache
- **SwiGLU** — gated FFN replacing GELU; better quality at same parameter count
- **RMSNorm** — faster normalisation (no mean-centering), used by Llama/Mistral/Gemma
- **Sliding Window Attention** — configurable window for O(n·window) memory
- **Parallel Attention+FFN** — PaLM-style shared norm input; ~15% faster training
- **Mixed Precision** — bfloat16 autocast on CUDA
- **Weight Tying** — embedding & output head share weights (saves ~38.6M params)
- **Z-Loss** — logit magnitude penalty (PaLM-style)
- **MoE** (optional) — top-2 routed experts; 510M total params, same FLOPs as 114M
- **AdamW** with linear warmup + cosine decay
- **Gradient accumulation** — effective batch = BATCH_SIZE × ACCUMULATION_STEPS
- **Text generation** — top-k, top-p, temperature sampling with KV cache

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Basic Model](#basic-model-usage)
  - [Training](#training)
  - [Text Generation](#text-generation)
  - [Training with Large Datasets](#training-with-large-datasets)
- [References](#references)
- [License](#license)

---

## Installation

```bash
git clone <repo-url>
cd kaito-model
pip install -r requirements.txt
```

### Dependencies

- `torch` (>=2.0) — PyTorch
- `tiktoken` — OpenAI's GPT-2 tokenizer
- `datasets` (optional) — HuggingFace datasets for large corpus training

---

## Project Structure

```
kaito-model/
├── config.py                  # All hyperparameters (globals, wildcard-imported)
├── main.py                    # kaitomodel class (model definition)
├── train.py                   # train_model function (AdamW, scheduler, loop)
├── text_generation.py         # generate_text with KV-cache & sampling
├── AGENTS.md                  # Repo conventions for AI coding assistants
├── data.md                    # Dataset recommendations & training guide
├── learning.ipynb             # Interactive training notebook
├── requirements.txt           # Python dependencies
├── model.pt                   # Saved checkpoint (gitignored)
│
├── model/
│   ├── architecture.py        # RMSNorm, TransformerBlock
│   ├── ffn.py                 # SwiGLU, GatedFeedForward (dense FFN)
│   └── moe.py                 # MoELayer (sparse expert routing)
│
├── multi_attention/
│   └── attention.py           # RotaryEmbedding, grouped_query_attention
│
├── data_prep/
│   ├── preprocess_text.py     # GPTDataset, PreprocessText
│   └── the-verdict.txt        # Default training data (-5K tokens)
│
├── loss/
│   └── cal_loss.py            # Cross-entropy + Z-loss, bfloat16 autocast
│
└── papers/                    # Reference papers (PDFs)
```

---

## Model Architecture

### Overview

The model is a **decoder-only transformer** with 12 blocks. Each block processes a
`(batch, seq_len, 768)` tensor through parallel attention + FFN branches and sums
into the residual stream (PaLM-style).

### Component Details

| Component | File | What it does |
|---|---|---|
| **RoPE** | `multi_attention/attention.py` | Applies rotary position to Q and K inside attention. No learned position embeddings needed — the model can extrapolate to arbitrary sequence lengths. |
| **GQA** | `multi_attention/attention.py` | 12 query heads share 4 KV heads (group size = 3). Reduces KV cache size by 3× with negligible quality loss (Ainslie et al., 2023). |
| **RMSNorm** | `model/architecture.py` | Normalises by root-mean-square (no mean-centering). ~20% faster than LayerNorm, empirically equivalent (Zhang & Sennrich, 2019). Used by Llama, Mistral, Gemma. |
| **SwiGLU** | `model/ffn.py` | Gated FFN: `output = W_down(Swish(W_gate(x)) * W_up(x))`. Matches standard FFN param count (hidden 2048 vs 3072) but outperforms GELU (Shazeer, 2020). |
| **MoE** (opt.) | `model/moe.py` | Replaces each FFN with 8 expert SwiGLU networks, top-2 routed. Adds load-balancing aux loss (`MOE_LOSS_COEFF`). Same FLOPs as dense path. |
| **Parallel FFN** | `model/architecture.py` | Attention and FFN share one normalised input and sum into the residual in a single step. ~15% faster training (Chowdhery et al., 2022). |
| **Sliding Window** | `multi_attention/attention.py` | Banded causal mask limiting each token to N previous tokens. O(n·window) memory instead of O(n²) (Mistral-style). |
| **Weight Tying** | `main.py:35` | `out_head.weight = token_embedding.weight`. Shared embedding/head saves ~38.6M params (Press & Wolf, 2017). |
| **Z-Loss** | `loss/cal_loss.py` | Penalty `1e-4 * logsumexp(logits)²` keeps logit magnitudes bounded. Without it, softmax logits drift during training (PaLM). |
| **Mixed Precision** | `loss/cal_loss.py` | bfloat16 autocast on CUDA (Ampere+ GPUs). ~2× throughput, identical numerical accuracy. No-op on CPU. |

### Forward Pass

```
input_ids (batch, seq_len)
    ↓
token_embedding → dropout
    ↓
for each of 12 TransformerBlock:
    normed = layernorm1(x)
    attn_out = grouped_query_attention(normed, past_keys, past_values)
    ffn_out  = feedforward(normed)
    x = x + attn_out + ffn_out
    ↓
final_norm (RMSNorm)
    ↓
out_head (Linear, weights tied with token_embedding)
    ↓
logits (batch, seq_len, vocab_size)
```

---

## Configuration

All hyperparameters live in `config.py` and are wildcard-imported as globals:

```python
# Model dimensions
VOCAB_SIZE = 50257       # GPT-2 tokeniser vocab
OUTPUT_DIM = 768         # d_model
N_HEADS = 12             # query heads
N_KV_HEADS = 4           # KV heads (GQA: 12/4, group=3)
N_LAYERS = 12            # transformer blocks
MAX_LENGTH = 512         # max sequence length

# Training
BATCH_SIZE = 2           # per-GPU batch
ACCUMULATION_STEPS = 4   # effective batch = BATCH_SIZE × ACCUMULATION_STEPS
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1       # AdamW
WARMUP_RATIO = 0.05      # fraction of total steps for LR warmup
GRAD_CLIP_MAX_NORM = 1.0
Z_LOSS_COEFF = 1e-4      # logit magnitude penalty

# MoE (off by default)
USE_MOE = False
N_EXPERTS = 8
MOE_TOP_K = 2
MOE_LOSS_COEFF = 0.01

# Generation
TEMPERATURE = 1.0
TOP_K = 50
TOP_P = 0.9
```

See `data.md` for dataset-specific tuning advice.

---

## Usage

### Basic Model

```python
from main import kaitomodel
import torch

model = kaitomodel()
x = torch.randint(0, 50257, (2, 16))
logits, kv_cache = model(x)       # returns tuple (logits, kv_cache)
print(logits.shape)                # (2, 16, 50257)
```

### Training

```python
from main import kaitomodel
from data_prep.preprocess_text import PreprocessText
from train import train_model
import tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

model = kaitomodel().to(device)
train_loader, val_loader = PreprocessText().preprocess()

train_losses, val_losses, tokens_seen = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=3,
    eval_freq=5,
    eval_iter=5,
    start_context="Hello",
    tokenizer=tokenizer,
)
# Model saved to model.pt automatically
```

The training loop internally creates:
- **AdamW** with weight decay on weight matrices only (not biases/norms)
- **Linear warmup + cosine decay** scheduler (stepped per optimizer step)
- **Gradient accumulation** (effective batch = BATCH_SIZE × ACCUMULATION_STEPS)
- **Gradient clipping** at GRAD_CLIP_MAX_NORM
- **MoE auxiliary loss** collection (if USE_MOE=True)

### Text Generation

```python
from text_generation import generate_text
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
model.eval()

result = generate_text(
    model=model,
    idx="In the beginning",
    tokenizer=tokenizer,
    new_max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)
print(result)
```

Generation uses **KV-cache** for O(n) per-step cost:
- Step 0: full prompt processed, KV cache populated
- Steps 1+: only the last token goes through attention (cached KVs reused)

### Training with Large Datasets

See `data.md` for:

- Recommended datasets (OpenWebText, FineWeb, WikiText-103, etc.)
- HuggingFace dataset integration with streaming
- Memory-mapped tokenisation for multi-GB corpora
- Hardware requirements & expected training times
- Quick-start commands for each dataset

---

## References

Research papers that inspired this implementation (included in `papers/`):

1. **Attention Is All You Need** — Vaswani et al., 2017
2. **GPT-2** — Radford et al., 2019
3. **GPT-3** — Brown et al., 2020
4. **Original GPT** — Radford et al., 2018
5. **InstructGPT** — Ouyang et al., 2022
6. **RoPE** — Su et al., 2021
7. **GQA** — Ainslie et al., 2023
8. **SwiGLU** — Shazeer, 2020
9. **RMSNorm** — Zhang & Sennrich, 2019
10. **PaLM** — Chowdhery et al., 2022 (parallel FFN, Z-loss)
11. **MoE** — Shazeer et al., 2017
12. **Weight Tying** — Press & Wolf, 2017

---

## License

This project is licensed under the terms in the LICENSE file.
