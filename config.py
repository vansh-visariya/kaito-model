## this is the config file for the kaito model (gpt2 small)

BATCH_SIZE =  2 # 32, used 2 due to less data
MAX_LENGTH = 512
STRIDE = 256
VOCAB_SIZE = 50257
OUTPUT_DIM = 768
N_HEADS = 12
N_LAYERS = 12
DROPOUT = 0.1
LEARNING_RATE = 0.0001

qkv_bias = False