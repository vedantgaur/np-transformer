import numpy as np
from attention import softmax

def cross_entropy_loss(y_pred, y_true):
    y_true = np.eye(y_pred.shape[-1])[y_true]
    log_softmax = np.log(softmax(y_pred))
    loss = -np.sum(y_true * log_softmax) / y_pred.shape[0]
    
    return loss

def softmax_grad(softmax_output, y_true):
    batch_size = y_true.shape[0]
    grad = softmax_output - y_true
    grad /= batch_size

    return grad


