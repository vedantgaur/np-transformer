import numpy as np

def ff(x, d_ff, d_model):
    W_1 = np.random.randn(d_model, d_ff)
    b_1 = np.random.randn(d_ff)

    x = x @ W_1 + b_1
    x = np.maximum(0, x)

    W_2 = np.random.randn(d_ff, d_model)
    b_2 = np.random.randn(d_model)
    x = x @ W_2 + b_2

    return x