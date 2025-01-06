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

def ff_backprop(x, d_ff, d_model, grad_output):
    W_1 = np.random.randn(d_model, d_ff)
    W_2 = np.random.randn(d_ff, d_model)

    grad_W_2 = x.T @ grad_output
    grad_b_2 = np.sum(grad_output, axis=0)

    grad_x = grad_output @ W_2.T
    
    grad_W_1 = x.T @ (grad_x * (x > 0))
    grad_b_1 = np.sum(grad_x * (x > 0), axis=0)

    return grad_W_1, grad_b_1, grad_W_2, grad_b_2, grad_x