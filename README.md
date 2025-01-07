# Transformer Implementation in NumPy

A from-scratch implementation of the Transformer architecture using only NumPy, designed for sequence-to-sequence tasks.

## Overview

This implementation includes:
- Full Transformer architecture (Encoder-Decoder)
- Multi-head attention
- Position-wise feedforward networks
- Layer normalization
- Positional encoding

## Project Structure
├── model/\
│ ├── transformer.py # Main Transformer implementation\
│ ├── encoder.py # Transformer encoder\
│ ├── decoder.py # Transformer decoder\
│ ├── attention.py # Multi-head attention mechanism\
│ └── feedforward.py # Position-wise feedforward network\
├── data/\
│ └── dataset.py # Dataset handling\
├── experiments/\
│ ├── train.py # Training loop\
│ └── test.py # Testing and evaluation\
├── utils/\
│ └── losses.py # Loss functions\
├── requirements.txt\
└── README.md\

## Installation
```
pip install -r requirements.txt
```
## Usage

### Training
```
python experiments/train.py
```
### Testing
```
python experiments/test.py
```

## Model Details

- Input: Sequence of token indices (e.g., text tokens)
- Output: Predicted next token in sequence
- Architecture:
  - Embedding dimension: 512
  - Feed-forward dimension: 2048
  - Number of heads: 8
  - Number of encoder/decoder layers: 6
  - Vocabulary size: 32000
  - Maximum sequence length: 512

## Data Format

The model expects:
- Source sequences: shape (batch_size, seq_length)
- Target sequences: shape (batch_size, seq_length + 1)

Each value should be an integer representing a token index in the vocabulary.

## Model Saving and Loading

Models can be saved and loaded using:

### Save model
```
np.save('model_weights.npy', model.get_weights())
```
### Load model
```
model.load_weights(np.load('model_weights.npy'))
```