## this is the config file for the kaito model (gpt2 small)

## Data / model
BATCH_SIZE =  2 # 32, used 2 due to less data
MAX_LENGTH = 512
STRIDE = 256
VOCAB_SIZE = 50257
OUTPUT_DIM = 768
N_HEADS = 12
N_KV_HEADS = 4  # GQA: 4 KV heads shared across 12 query heads (3:1 ratio)
N_LAYERS = 12
DROPOUT = 0.1
LEARNING_RATE = 0.0001

qkv_bias = False

## Training — gradient scaling, regularisation & LR schedule
ACCUMULATION_STEPS = 4    # gradient accumulation: simulates BATCH_SIZE*4 effective batch
GRAD_CLIP_MAX_NORM = 1.0  # max gradient norm for clipping (prevents gradient explosion)
WEIGHT_DECAY = 0.1        # AdamW decoupled weight decay (0.1 is standard for LLMs)
WARMUP_RATIO = 0.05       # fraction of total optimizer steps for LR warmup
Z_LOSS_COEFF = 1e-4       # auxiliary loss penalising logit magnitudes (PaLM-style)

## Generation — sampling defaults
TEMPERATURE = 1.0    # lower = sharper (0=argmax); higher = more random
TOP_K = 50           # 0 = disabled; 50 means sample from top-50 tokens only (arXiv:1805.04833)
TOP_P = 0.9          # 1.0 = disabled; cumulative probability threshold for nucleus sampling