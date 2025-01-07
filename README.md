# Transformer Implementation in NumPy

A from-scratch implementation of the Transformer architecture using only NumPy, designed for sequence-to-sequence tasks.

## Overview

This implementation includes:
- Full Transformer architecture (Encoder-Decoder)
- Multi-head attention
- Position-wise feedforward networks
- Layer normalization
- Positional encoding

Certainly! Here's a cleaner and more readable version of the project structure for the README:

---

## Project Structure

The project is organized as follows:

```
├── model/                          # Core Transformer Model Implementation
│   ├── transformer.py              # Main Transformer model
│   ├── encoder.py                 # Transformer encoder
│   ├── decoder.py                 # Transformer decoder
│   ├── attention.py               # Multi-head attention mechanism
│   └── feedforward.py             # Position-wise feedforward network
│
├── data/                           # Dataset Handling
│   └── dataset.py                 # Dataset loading and processing
│
├── experiments/                    # Experiment-related scripts
│   ├── train.py                   # Training loop
│   └── test.py                    # Testing and evaluation
│
├── utils/                          # Utility functions
│   └── losses.py                  # Loss functions
│
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

### Directory Descriptions:

- **model/**: Contains the core components of the Transformer model including the encoder, decoder, attention mechanism, and feedforward layers.
- **data/**: Handles loading and preprocessing of datasets for training and evaluation.
- **experiments/**: Contains scripts to run experiments like training and testing.
- **utils/**: Includes utility functions such as custom loss functions.
- **requirements.txt**: Lists the Python dependencies required to run the project.
- **README.md**: Project overview and documentation.

---

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