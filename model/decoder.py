import numpy as np

from model.attention import self_attention, cross_attention, attention_backprop
from model.encoder import layer_norm, layer_norm_backward
from model.feedforward import ff, ff_backprop

def multi_layer_decoder(x, enc_output, d_ff, d_model, h, num_layers):
    for _ in range(num_layers):
        x = decoder(x, enc_output, d_ff, d_model, h)
    return x

def decoder(x, enc_output, d_ff, d_model, h):
    self_attn = self_attention(x, d_model, h)
    layer_norm_self_attn = layer_norm(x+self_attn)

    cross_attn = cross_attention(layer_norm_self_attn, enc_output, d_model, h)
    layer_norm_cross_attn = layer_norm(layer_norm_self_attn+cross_attn)

    ffn_output = ff(layer_norm_cross_attn, d_ff, d_model)
    layer_norm_ffn = layer_norm(layer_norm_cross_attn+ffn_output)

    return layer_norm_ffn

def decoder_backward(grad_output, d_ff, d_model, h):
    grad_ff_weights = ff_backprop(grad_output, d_ff, d_model, grad_output)
    grad_pre_ff = grad_ff_weights[-1]
    
    grad_cross = layer_norm_backward(grad_pre_ff)
    
    W_Q = np.random.randn(d_model, d_model) * 0.02
    W_K = np.random.randn(d_model, d_model) * 0.02
    W_V = np.random.randn(d_model, d_model) * 0.02
    
    Q = grad_cross @ W_Q
    K = grad_cross @ W_K
    V = grad_cross @ W_V
    
    grad_Q, grad_K, grad_V = attention_backprop(Q, K, V, grad_cross, h, d_model)
    
    grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(grad_cross.shape[0], grad_cross.shape[1], d_model)
    grad_K = grad_K.transpose(0, 2, 1, 3).reshape(grad_cross.shape[0], grad_cross.shape[1], d_model)
    grad_V = grad_V.transpose(0, 2, 1, 3).reshape(grad_cross.shape[0], grad_cross.shape[1], d_model)
    
    grad_decoder = grad_Q @ W_Q.T
    grad_encoder = (grad_K @ W_K.T + grad_V @ W_V.T) / 2
    
    grad_self = layer_norm_backward(grad_decoder)
    
    Q = grad_self @ W_Q
    K = grad_self @ W_K
    V = grad_self @ W_V
    
    grad_Q, grad_K, grad_V = attention_backprop(Q, K, V, grad_self, h, d_model)
    
    # Reshape gradients again before final multiplication
    grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(grad_self.shape[0], grad_self.shape[1], d_model)
    grad_K = grad_K.transpose(0, 2, 1, 3).reshape(grad_self.shape[0], grad_self.shape[1], d_model)
    grad_V = grad_V.transpose(0, 2, 1, 3).reshape(grad_self.shape[0], grad_self.shape[1], d_model)
    
    grad_x = (grad_Q @ W_Q.T + grad_K @ W_K.T + grad_V @ W_V.T) / 3
    
    return grad_x, grad_encoder

