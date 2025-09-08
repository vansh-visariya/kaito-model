# Kaito Model: GPT-2 Implementation from Scratch

A complete implementation of a GPT-2 style transformer model built from scratch using PyTorch. This project demonstrates the core concepts of transformer architecture, self-attention mechanisms, and autoregressive language modeling. [model](https://huggingface.co/vansh-myth/kaito)

## 🚀 Features

- **Complete GPT-2 Architecture**: Full implementation including multi-head attention, feed-forward networks, and layer normalization
- **Custom Attention Mechanism**: Detailed implementation of masked multi-head self-attention
- **Text Generation**: Autoregressive text generation capabilities
- **Training Pipeline**: Complete training loop with loss calculation and evaluation
- **Modular Design**: Clean, well-documented code structure for educational purposes

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training](#training)
- [Text Generation](#text-generation)
- [Understanding the Workflow](#understanding-the-workflow)
- [References](#references)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- `torch` - PyTorch deep learning framework
- `tiktoken` - OpenAI's tokenizer
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Plotting and visualization
- `jupyterlab` - Interactive development environment

## 📁 Project Structure

```
llm-from-scratch/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Model configuration parameters
├── main.py                      # Main model class (KaitoModel)
├── train.py                     # Training and evaluation functions
├── text_generation.py           # Text generation utilities
├── learning.ipynb              # Jupyter notebook for experiments
├── model.pt                     # Saved model weights
├── LICENSE                      # License file
│
├── model/
│   └── architecture.py         # Core model components (LayerNorm, GELU, FeedForward, TransformerBlock)
│
├── multi_attention/
│   └── attention.py            # Multi-head self-attention implementation
│
├── data_prep/
│   ├── preprocess_text.py      # Text preprocessing and dataset creation
│   └── the-verdict.txt         # Training data
│
├── loss/
│   └── cal_loss.py             # Loss calculation functions
│
└── papers/                     # Reference papers
    ├── attention_is_all_you_need.pdf
    ├── language_models_are_unsupervised_multitask_learners.pdf
    ├── Language Models are Few-Shot Learners.pdf
    ├── language_understanding_gpt.pdf
    └── train_llm_on_instruction.pdf
```

## 🏗️ Model Architecture

The Kaito model implements a GPT-2 style decoder-only transformer with the following components:

### Core Components

1. **Token & Positional Embeddings** (`main.py`)
   - Token embedding: Maps vocabulary indices to dense vectors
   - Positional embedding: Learned positional encodings
   - Combined embedding: Token + positional embeddings

2. **Multi-Head Self-Attention** (`multi_attention/attention.py`)
   - Scaled dot-product attention with causal masking
   - Multiple attention heads for parallel processing
   - Query, Key, Value projections with optional bias

3. **Feed-Forward Network** (`model/architecture.py`)
   - Two-layer MLP with GELU activation
   - Dimension expansion (768 → 3072 → 768)
   - Dropout for regularization

4. **Layer Normalization** (`model/architecture.py`)
   - Pre-normalization before attention and FFN
   - Learnable scale and shift parameters
   - Stabilizes training and improves convergence

5. **Transformer Block** (`model/architecture.py`)
   - Combines attention, FFN, and residual connections
   - Pre-layer normalization architecture
   - Dropout for regularization

## ⚙️ Configuration

**Model Configuration** (defined in `config.py`):
* `BATCH_SIZE` = 2 (reduced due to limited data)
* `MAX_LENGTH` = 512 (maximum sequence length)
* `STRIDE` = 256 (sliding window stride for data preparation)
* `VOCAB_SIZE` = 50257 (GPT-2 vocabulary size)
* `OUTPUT_DIM` = 768 (model dimension)
* `N_HEADS` = 12 (number of attention heads)
* `N_LAYERS` = 12 (number of transformer blocks)
* `DROPOUT` = 0.1 (dropout rate)
* `LEARNING_RATE` = 0.0001 (learning rate for training)
* `qkv_bias` = False (whether to use bias in QKV projections)

## 🚀 Usage

### Basic Model Usage

```python
from main import kaitomodel
from config import *
import torch

# Initialize model
model = kaitomodel()

# Example input (batch_size=2, seq_len=10)
input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))

# Forward pass
logits = model(input_ids)  # Shape: (2, 10, 50257)
```

### Data Preprocessing

```python
from data_prep.preprocess_text import PreprocessText

# Initialize preprocessor
preprocessor = PreprocessText("data_prep/the-verdict.txt")

# Create dataloaders
train_loader, val_loader = preprocessor.preprocess()
```

## 🎯 Training

The training pipeline includes:

1. **Loss Calculation**: Cross-entropy loss for next-token prediction
2. **Evaluation**: Separate validation loss tracking
3. **Text Generation**: Periodic generation during training for monitoring

```python
from train import train_model
from main import kaitomodel
import torch.optim as optim

# Initialize model and optimizer
model = kaitomodel()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train model
train_losses, val_losses, tokens_seen = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=10,
    eval_freq=100,
    eval_iter=50,
    start_context="Hello",
    tokenizer=tokenizer
)
```

## 📝 Text Generation

The model supports autoregressive text generation:

```python
from text_generation import generate_text
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

# Generate text
generated = generate_text(
    model=model,
    prompt="Hello, world",
    tokenizer=tokenizer,
    new_max_length=50
)
print(generated)
```

## 🔄 Understanding the Workflow

### **Starting Point: The Input Tensor**

The workflow begins after tokenization and embedding. You have a single tensor that is the sum of your token embeddings and positional embeddings.

* **Shape:** `(BATCH_SIZE, MAX_LENGTH, OUTPUT_DIM)` or `(2, 512, 768)`

This tensor represents your entire batch of text sequences, ready to be processed by the Transformer.

### **The Core Workflow: The Stack of Decoder Blocks**

The main body of the GPT-2 model is a stack of 12 identical decoder blocks. The tensor passes through these 12 blocks one by one. The output of Block 1 becomes the input to Block 2, and so on. The shape of the tensor **does not change** as it moves through this stack.

Here's what happens inside a **single decoder block**:

1.  **Pre-Layer Normalization (First)**
    * The `(2, 512, 768)` tensor is first normalized. Layer Normalization stabilizes the data for the next step by ensuring the inputs to the attention layer have a consistent distribution.

2.  **Masked Multi-Head Self-Attention**
    * The normalized tensor is used to generate the Query, Key, and Value matrices.
    * The model calculates attention scores, but with the crucial **causal mask** applied. This mask prevents any token from "seeing" or gathering information from future tokens in the sequence. This is the core of auto-regressive text generation.
    * The output is a new `(2, 512, 768)` tensor where each token's vector is now context-aware, containing information from itself and all previous tokens.

3.  **Residual (Skip) Connection (First)**
    * The original input to the block (from before the first LayerNorm) is **added** to the output of the attention layer. This is a critical step that allows gradients to flow easily during training and prevents the model from losing the original information.

4.  **Pre-Layer Normalization (Second)**
    * The result of the skip connection is normalized again to prepare it for the next sub-layer.

5.  **Feed-Forward Network (FFN)**
    * The normalized tensor is passed through a two-layer neural network.
    * This network typically expands the dimension (e.g., from 768 to 3072) and then contracts it back down (from 3072 to 768), with a GELU activation in between.
    * This is where the model performs much of its "computation" or "reasoning" on the contextual information gathered by the attention mechanism.

6.  **Residual (Skip) Connection (Second)**
    * The input to the FFN (from before the second LayerNorm) is **added** to the output of the FFN.

After this final step, the `(2, 512, 768)` tensor exits the current decoder block and is passed as input to the next one. This entire process is repeated 12 times.

### **Final Step: The Language Model Head**

After the tensor has passed through all 12 decoder blocks, it undergoes two final transformations to become the prediction.

1.  **Final Layer Normalization**
    * The output from the 12th block is normalized one last time.

2.  **Projection to Logits**
    * This is the most important step for getting the final prediction. A final linear layer, often called the "language model head," is applied.
    * This layer's job is to project the final high-dimensional vector (`d_model` = 768) for each token into a much larger vector the size of your vocabulary (`VOCAB_SIZE` = 50257).
    * **Shape Change**: `(2, 512, 768)` → `(2, 512, 50257)`

### **The Final Output: Logits**

The final tensor, with the shape `(2, 512, 50257)`, is your **logits** tensor.

* For each of the 2 sequences in your batch, and for each of the 512 token positions, you now have a vector of 50,257 raw, un-normalized scores. Each score represents the model's prediction for how likely that word is to be the next token.

To perform next-token prediction, you would typically take the logits for the very last token in your input sequence (e.g., at position 511), apply a softmax function to convert them into probabilities, and then sample from that distribution to generate the next word.

## 🧠 Key Implementation Details

### Self-Attention Mechanism

The multi-head self-attention implementation includes several important concepts:

**Query, Key, Value Concept:**
- **Query**: What a word wants to know
- **Key**: What a word offers
- **Value**: The actual information

**Attention Process:**
1. `Scores = Q·K^T` (Find relevance between tokens)
2. `scaled_score = Scores/√d_k` (Normalize scores)
3. `attention_weights = softmax(scaled_score)` (Convert to probabilities)
4. `output = attention_weights·V` (Weighted sum of values)

**Causal Masking:**
- Prevents tokens from attending to future positions
- Essential for autoregressive generation
- Implemented using upper triangular mask

### Layer Normalization vs Batch Normalization

**Layer Normalization** (used in this model):
- Normalizes each sample individually across the feature dimension (row-level)
- Best for RNNs and Transformers
- Stable across different batch sizes

**Batch Normalization** (not used):
- Normalizes each feature across the batch dimension (column-level)
- Best for CNNs
- Can be unstable with small batch sizes

### GELU vs ReLU Activation

**GELU** (used in this model):
- More smooth and continuous than ReLU
- For small negative values, outputs small negative values instead of zero
- Helps the model learn better representations

**ReLU** (not used):
- Hard cutoff at zero for negative values
- Can cause dead neurons

### Positional Encoding Options

The project includes documentation for various positional encoding methods:

1. **Learned Positional Embedding** (used in this model)
   - Initialize with random values and learn during training
   - Simple and effective for fixed sequence lengths

2. **Sinusoidal Positional Embedding**
   - Uses sine and cosine functions with different frequencies
   - Provides unique relative positions
   - Good for variable sequence lengths

3. **Rotary Positional Embedding (RoPE)**
   - Rotates Query and Key vectors based on position
   - Relative position information in attention computation

4. **Relative Positional Embedding**
   - Attention scores depend on relative distance, not absolute position
   - Adds bias term based on relative distance

## 📊 Loss and Evaluation

### Cross-Entropy Loss
- Used for next-token prediction (classification task)
- Formula: `-log(p(y|x))` where `p(y|x)` is the probability of target given input

### Perplexity
- Measures model uncertainty
- Formula: `e^cross_entropy`
- Lower perplexity = better model
- Perplexity of 10 means model is as confused as randomly choosing between 10 words

## 🔧 Data Processing Pipeline

### Text Preprocessing Steps:
1. **Load text** from file
2. **Clean text** using regex (remove symbols, keep words)
3. **Tokenize** using GPT-2 tokenizer (tiktoken)
4. **Create sliding windows** with stride for training sequences
5. **Generate dataloaders** with train/validation split

### Dataset Creation:
- Uses sliding window approach with configurable stride
- Input sequence: `[0,1,2,3]`
- Target sequence: `[1,2,3,4]` (shifted by one position)
- Enables next-token prediction training

## 🎮 Getting Started

### Quick Start Example:

```python
# 1. Import necessary modules
from main import kaitomodel
from data_prep.preprocess_text import PreprocessText
from train import train_model
import torch
import torch.optim as optim
import tiktoken

# 2. Initialize components
model = kaitomodel()
preprocessor = PreprocessText()
tokenizer = tiktoken.get_encoding("gpt2")

# 3. Prepare data
train_loader, val_loader = preprocessor.preprocess()

# 4. Set up training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 5. Train the model
train_losses, val_losses, tokens_seen = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=5,
    eval_freq=50,
    eval_iter=25,
    start_context="The",
    tokenizer=tokenizer
)

# 6. Generate text
from text_generation import generate_text
generated_text = generate_text(
    model=model,
    prompt="Once upon a time",
    tokenizer=tokenizer,
    new_max_length=100
)
print(f"Generated: {generated_text}")
```

## 📚 References

This implementation is based on the following research papers (included in `papers/` directory):

1. **"Attention Is All You Need"** - Original Transformer paper
2. **"Language Models are Unsupervised Multitask Learners"** - GPT-2 paper
3. **"Language Models are Few-Shot Learners"** - GPT-3 paper
4. **"Improving Language Understanding by Generative Pre-Training"** - Original GPT paper
5. **"Training Language Models to Follow Instructions"** - InstructGPT paper

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- OpenAI for the GPT-2 architecture and tiktoken tokenizer
- The PyTorch team for the excellent deep learning framework
- The research community for the foundational papers on transformer architecture

---

**Note**: This is an educational implementation designed to understand transformer architecture. For production use, consider using established libraries like Hugging Face Transformers.
