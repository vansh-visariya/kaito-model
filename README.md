# kaito Workflow: A Step-by-Step Guide

**Configuration:**
* `BATCH_SIZE` = 2
* `MAX_LENGTH` = 512
* `OUTPUT_DIM` (`d_model`) = 768
* `N_LAYERS` = 12

---
### **Starting Point: The Input Tensor**

The workflow begins after tokenization and embedding. You have a single tensor that is the sum of your token embeddings and positional embeddings.

* **Shape:** `(BATCH_SIZE, MAX_LENGTH, OUTPUT_DIM)` or `(2, 512, 768)`

This tensor represents your entire batch of text sequences, ready to be processed by the Transformer.

---
### **The Core Workflow: The Stack of Decoder Blocks**

The main body of the GPT-2 model is a stack of 12 identical decoder blocks. The tensor passes through these 12 blocks one by one. The output of Block 1 becomes the input to Block 2, and so on. The shape of the tensor **does not change** as it moves through this stack.

Here’s what happens inside a **single decoder block**:



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

---
### **Final Step: The Language Model Head**

After the tensor has passed through all 12 decoder blocks, it undergoes two final transformations to become the prediction.

1.  **Final Layer Normalization**
    * The output from the 12th block is normalized one last time.

2.  **Projection to Logits**
    * This is the most important step for getting the final prediction. A final linear layer, often called the "language model head," is applied.
    * This layer's job is to project the final high-dimensional vector (`d_model` = 768) for each token into a much larger vector the size of your vocabulary (`VOCAB_SIZE` = 50257).
    * **Shape Change**: `(2, 512, 768)` → `(2, 512, 50257)`

---
### **The Final Output: Logits**

The final tensor, with the shape `(2, 512, 50257)`, is your **logits** tensor.

* For each of the 2 sequences in your batch, and for each of the 512 token positions, you now have a vector of 50,257 raw, un-normalized scores. Each score represents the model's prediction for how likely that word is to be the next token.

To perform next-token prediction, you would typically take the logits for the very last token in your input sequence (e.g., at position 511), apply a softmax function to convert them into probabilities, and then sample from that distribution to generate the next word.