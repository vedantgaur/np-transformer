import numpy as np
from encoder import multi_layer_encoder
from decoder import multi_layer_decoder

# Define the model parameters
d_model = 16
d_ff = 64
h = 4
num_layers = 2

# Generate some random test data for x and enc_output
x = np.random.randn(8, d_model)  # Batch size of 8, d_model features
enc_output = np.random.randn(8, d_model)  # Encoder output (same shape as x)

# Test the encoder
enc_output_test = multi_layer_encoder(x, d_ff, d_model, h, num_layers)
print("Encoder output shape:", enc_output_test.shape)

# Test the decoder
decoder_output_test = multi_layer_decoder(x, enc_output, d_ff, d_model, h, num_layers)
print("Decoder output shape:", decoder_output_test.shape)
