import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dp_attn(Q, K, V, mask=None):
    dp = Q @ K.transpose(0, 2, 1)
    scaled_dp = dp / np.sqrt(np.shape(K)[-1])

    if mask is not None:
        scaled_dp += mask
    
    attn_weights = softmax(scaled_dp)
    return attn_weights @ V

def multi_headed_attn(Q, K, V, h: int, d_model: int, mask=None):
    batch_size, seq_len, d_k = Q.shape
    d_v = V.shape[-1]

    assert d_k % h == 0, f"d_k ({d_k}) must be divisible by h ({h})"
    assert d_v % h == 0, f"d_v ({d_v}) must be divisible by h ({h})"

    Q_split = Q.reshape(batch_size, seq_len, h, d_k // h).transpose(0, 2, 1, 3)
    K_split = K.reshape(batch_size, seq_len, h, d_k // h).transpose(0, 2, 1, 3)
    V_split = V.reshape(batch_size, seq_len, h, d_v // h).transpose(0, 2, 1, 3)

    output = np.zeros((batch_size, h, seq_len, d_v // h))

    for i in range(h):
        output[:, i] = scaled_dp_attn(Q_split[:, i], K_split[:, i], V_split[:, i], mask)
    
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_v)

    W_0 = np.random.randn(d_v, d_model) * 0.02

    return output @ W_0

def self_attention(x, d_model, h):
    _, seq_len, _ = x.shape
    
    W_Q = np.random.randn(d_model, d_model) * 0.02
    W_K = np.random.randn(d_model, d_model) * 0.02
    W_V = np.random.randn(d_model, d_model) * 0.02

    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V

    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e10

    return multi_headed_attn(Q, K, V, h, d_model, mask)

def cross_attention(x, enc_output, d_model, h):
    W_Q = np.random.randn(d_model, d_model) * 0.02
    W_K = np.random.randn(d_model, d_model) * 0.02
    W_V = np.random.randn(d_model, d_model) * 0.02

    Q = x @ W_Q
    K = enc_output @ W_K
    V = enc_output @ W_V

    return multi_headed_attn(Q, K, V, h, d_model)

def attention_backprop(Q, K, V, grad_attention, h, d_model):
    batch_size = Q.shape[0]
    seq_len = Q.shape[1]
    
    dp = Q @ K.transpose(0, 2, 1)
    scaled_dp = dp / np.sqrt(K.shape[-1])
    
    exp_scaled_dp = np.exp(scaled_dp - np.max(scaled_dp, axis=-1, keepdims=True))
    softmax_output = exp_scaled_dp / np.sum(exp_scaled_dp, axis=-1, keepdims=True)
    
    grad_softmax = softmax_output * (grad_attention @ V.transpose(0, 2, 1) - 
                                   np.sum((grad_attention @ V.transpose(0, 2, 1)) * softmax_output, 
                                        axis=-1, keepdims=True))
    
    grad_dp = grad_softmax / np.sqrt(K.shape[-1])
    
    grad_Q = grad_dp @ K
    grad_K = grad_dp.transpose(0, 2, 1) @ Q
    grad_V = softmax_output.transpose(0, 2, 1) @ grad_attention
    
    head_size = d_model // h
    grad_Q = grad_Q.reshape(batch_size, h, seq_len, head_size)
    grad_K = grad_K.reshape(batch_size, h, seq_len, head_size)
    grad_V = grad_V.reshape(batch_size, h, seq_len, head_size)
    
    return grad_Q, grad_K, grad_V

if __name__ == "__main__":
    batch_size = 1
    seq_length = 5
    d_model = 8
    h = 2

    x = np.random.randn(batch_size, seq_length, d_model)
    mask = np.triu(np.ones((seq_length, seq_length)), k=1) * -1e10
    Q = K = V = x

    attention_output = multi_headed_attn(Q, K, V, h, d_model, mask)
    print("Attention Output Shape:", attention_output.shape) 