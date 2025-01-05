import numpy as np

def scaled_dp_attn(Q, K, V):
    dp = Q @ K.transpose(0, 2, 1)
    scaled_dp = dp / np.sqrt(np.shape(K)[-1])
    softmax = np.exp(scaled_dp)/np.sum(np.exp(scaled_dp), axis=-1, keepdims=True)
    # print(np.sum(softmax, axis=-1))

    return softmax @ V

def multi_headed_attn(Q, K, V, h: int, d_model: int):
    batch_size, seq_len, d_k = Q.shape
    d_v = V.shape[-1]

    Q_split = Q.reshape(batch_size, seq_len, h, d_k // h).transpose(0, 2, 1, 3)
    K_split = K.reshape(batch_size, seq_len, h, d_k // h).transpose(0, 2, 1, 3)
    V_split = V.reshape(batch_size, seq_len, h, d_v // h).transpose(0, 2, 1, 3)

    output = []
    for i in range(h):
        ith_attn = scaled_dp_attn(Q_split[:, i], K_split[:, i], V_split[:, i])
        output.append(ith_attn)
    
    output = np.concatenate(output, axis=-1)

    W_0 = np.random.randn(h * d_v, d_model)
    return output @ W_0    

if __name__ == "__main__":
    batch_size = 1
    seq_length = 5
    d_model = 8 
    d_k = 8
    d_v = 8

    Q = np.random.randn(batch_size, seq_length, d_k)
    K = np.random.randn(batch_size, seq_length, d_k)
    V = np.random.randn(batch_size, seq_length, d_v)
    # print(scaled_dp_attn(Q, K, V))
    # print(np.sum(scaled_dp_attn(Q, K, V)[0][0]))