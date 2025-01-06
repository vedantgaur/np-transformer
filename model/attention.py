import numpy as np

def scaled_dp_attn(Q, K, V, mask=None):
    dp = Q @ K.transpose(0, 2, 1)
    scaled_dp = dp / np.sqrt(np.shape(K)[-1])
    print("Scaled DP (before applying mask):")
    print(scaled_dp)

    if mask is not None:
        scaled_dp += mask
        print("Scaled DP (after applying mask):")
        print(scaled_dp)
    
    softmax = np.exp(scaled_dp - np.max(scaled_dp)) / np.sum(np.exp(scaled_dp - np.max(scaled_dp)), axis=-1, keepdims=True)
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