import numpy as np

from attention import multi_headed_attn
from feedforward import ff

def multi_layer_encoder(x, d_ff, d_model, h, num_layers):
    for _ in range(num_layers):
        x = encoder(x, d_ff, d_model, h)
    return x

def encoder(x, d_ff, d_model, h):
    batch_size = x.shape[0]
    seq_len = 1

    x = x.reshape(batch_size, seq_len, d_model)

    W_Q = np.random.randn(d_model, d_model // h)
    W_K = np.random.randn(d_model, d_model // h)
    W_V = np.random.randn(d_model, d_model // h)

    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V

    attn = multi_headed_attn(Q, K, V, h, d_model)
    layer_norm_attn = layer_norm(x + attn)

    ffn_output = ff(layer_norm_attn, d_ff, d_model)
    
    layer_norm_ffn = layer_norm(layer_norm_attn + ffn_output)

    return layer_norm_ffn

def layer_norm(x, gamma=1, beta=0):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + 1e-8)
    return x_norm*gamma + beta