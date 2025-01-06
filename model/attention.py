import numpy as np

def softmax(x):
    np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=-1, keepdims=True)

def scaled_dp_attn(Q, K, V, mask=None):
    dp = Q @ K.transpose(0, 2, 1)
    scaled_dp = dp / np.sqrt(np.shape(K)[-1])
    print("Scaled DP (before applying mask):")
    print(scaled_dp)

    if mask is not None:
        scaled_dp += mask
        print("Scaled DP (after applying mask):")
        print(scaled_dp)
    
    softmax = softmax(scaled_dp)
    print("Softmax:")
    print(softmax)


    return softmax @ V

def multi_headed_attn(Q, K, V, h: int, d_model: int, mask=None):
    batch_size, seq_len, d_k = Q.shape
    d_v = V.shape[-1]

    Q_split = Q.reshape(batch_size, seq_len, h, d_k // h).transpose(0, 2, 1, 3)
    K_split = K.reshape(batch_size, seq_len, h, d_k // h).transpose(0, 2, 1, 3)
    V_split = V.reshape(batch_size, seq_len, h, d_v // h).transpose(0, 2, 1, 3)

    output = []
    for i in range(h):
        ith_attn = scaled_dp_attn(Q_split[:, i], K_split[:, i], V_split[:, i], mask)
        output.append(ith_attn)
    
    output = np.concatenate(output, axis=-1)

    W_0 = np.random.randn(h * (d_k // h), d_model)

    return output @ W_0    

def self_attention(x, d_model, h):
    W_Q = np.random.randn(d_model, d_model // h)
    W_K = np.random.randn(d_model, d_model // h)
    W_V = np.random.randn(d_model, d_model // h)

    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V

    mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1) * -1e10

    return multi_headed_attn(Q, K, V, h, d_model, mask)

def cross_attention(x, enc_output, d_model, h):
    W_Q = np.random.randn(d_model, d_model // h)
    W_K = np.random.randn(d_model, d_model // h)
    W_V = np.random.randn(d_model, d_model // h)

    Q = x @ W_Q
    K = enc_output @ W_K
    V = enc_output @ W_V

    return multi_headed_attn(Q, K, V, h, d_model)

def attention_backprop(Q, K, V, grad_attention, h, d_model):
    dp = Q @ K.transpose(0, 2, 1)
    scaled_dp = dp / np.sqrt(K.shape[-1])
    softmax_output = np.exp(scaled_dp) / np.sum(np.exp(scaled_dp), axis=-1, keepdims=True)

    grad_softmax = np.diagflat(softmax_output) - np.outer(softmax_output, softmax_output)

    grad_attention = grad_attention @ grad_softmax

    grad_V = grad_attention.T @ Q
    grad_K = grad_attention.T @ K
    grad_Q = grad_attention @ V

    grad_Q = grad_Q.reshape(h, grad_Q.shape[0], grad_Q.shape[1] // h).transpose(0, 2, 1)
    grad_K = grad_K.reshape(h, grad_K.shape[0], grad_K.shape[1] // h).transpose(0, 2, 1)
    grad_V = grad_V.reshape(h, grad_V.shape[0], grad_V.shape[1] // h).transpose(0, 2, 1)

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