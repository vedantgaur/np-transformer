import numpy as np

from model.attention import multi_headed_attn
from model.feedforward import ff, ff_backprop
from model.attention import attention_backprop

def multi_layer_encoder(x, d_ff, d_model, h, num_layers):
    for _ in range(num_layers):
        x = encoder(x, d_ff, d_model, h)
    return x

def encoder(x, d_ff, d_model, h):
    batch_size, seq_len, _ = x.shape

    x = x.reshape(batch_size, seq_len, d_model)

    W_Q = np.random.randn(d_model, d_model)
    W_K = np.random.randn(d_model, d_model)
    W_V = np.random.randn(d_model, d_model)

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

def encoder_backward(grad_output, d_ff, d_model, h):
    grad_ff_weights = ff_backprop(grad_output, d_ff, d_model, grad_output)
    grad_pre_ff = grad_ff_weights[-1]
    
    grad_ln = layer_norm_backward(grad_pre_ff)
    
    batch_size = grad_ln.shape[0]
    
    W_Q = np.random.randn(d_model, d_model) * 0.02
    W_K = np.random.randn(d_model, d_model) * 0.02
    W_V = np.random.randn(d_model, d_model) * 0.02
    
    Q = grad_ln @ W_Q
    K = grad_ln @ W_K
    V = grad_ln @ W_V
    
    grad_Q, grad_K, grad_V = attention_backprop(Q, K, V, grad_ln, h, d_model)
    
    grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, grad_ln.shape[1], d_model)
    grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, grad_ln.shape[1], d_model)
    grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, grad_ln.shape[1], d_model)
    
    grad_x = (grad_Q @ W_Q.T + grad_K @ W_K.T + grad_V @ W_V.T) / 3
    
    return grad_x

def layer_norm_backward(grad_output, gamma=1):
    mean = np.mean(grad_output, axis=-1, keepdims=True)
    var = np.var(grad_output, axis=-1, keepdims=True)
    
    grad_x = gamma * (grad_output - mean) / np.sqrt(var + 1e-8)
    
    return grad_x