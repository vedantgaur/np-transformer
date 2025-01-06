import numpy as np
from encoder import multi_layer_encoder
from decoder import multi_layer_decoder

# Define model parameters
d_model = 16  # Model dimension
d_ff = 64     # Feedforward network dimension
h = 4         # Number of attention heads
num_layers = 2  # Number of encoder/decoder layers
seq_len = 10  # Sequence length
batch_size = 8  # Batch size

# Generate test input data
x = np.random.randn(batch_size, seq_len, d_model)  # Input sequence
enc_output = np.random.randn(batch_size, seq_len, d_model)  # Encoder output for decoder testing

# Test the encoder
print("Testing Encoder...")
enc_output_test = multi_layer_encoder(x, d_ff, d_model, h, num_layers)
print("Encoder output shape:", enc_output_test.shape)
assert enc_output_test.shape == (batch_size, seq_len, d_model), "Encoder output shape mismatch"
# Test the decoder
print("\nTesting Decoder...")
decoder_output_test = multi_layer_decoder(x, enc_output, d_ff, d_model, h, num_layers)
print("Decoder output shape:", decoder_output_test.shape)
assert decoder_output_test.shape == (batch_size, seq_len, d_model), "Decoder output shape mismatch"

print("\nAll tests passed!")
