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

def ff_backprop(grad_output, d_ff, d_model, x):
    W_1 = np.random.randn(d_model, d_ff) * 0.02
    W_2 = np.random.randn(d_ff, d_model) * 0.02
    
    hidden = x @ W_1
    hidden_activated = np.maximum(0, hidden)
    output = hidden_activated @ W_2
    
    grad_hidden_activated = grad_output @ W_2.T
    grad_hidden = grad_hidden_activated * (hidden > 0)
    
    grad_W_2 = hidden_activated.transpose(0, 2, 1) @ grad_output
    grad_b_2 = np.sum(grad_output, axis=(0, 1))
    
    grad_W_1 = x.transpose(0, 2, 1) @ grad_hidden
    grad_b_1 = np.sum(grad_hidden, axis=(0, 1))
    
    grad_x = grad_hidden @ W_1.T
    
    return grad_W_1, grad_b_1, grad_W_2, grad_b_2, grad_x