import numpy as np

from attention import self_attention, cross_attention
from encoder import layer_norm
from feedforward import ff

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

