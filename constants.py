# -*- coding: utf-8 -*-

##
# Network
##

# Size of LSTM cell hidden states and word embeddings.
CELL_SIZE = 256
# Number of LSTM layers
NUM_LAYERS = 2

##
# Training
##

# Learning rate for training.
L_RATE = 0.002
# Number of sequences in a training batch.
BATCH_SIZE = 50
# Length of sequence to train on,
# eg., seq_len=5 => inputs='The quick brown fox jumps', targets='quick brown fox jumps over'.
SEQ_LEN = 5

# Directory in which to save trained models.
MODEL_SAVE_DIR = ''

# How often to save the model, in # steps.
MODEL_SAVE_FREQ = 5000
