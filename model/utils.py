import numpy as np

def cross_entropy_loss(logits, targets):
    batch_size = logits.shape[0]
    log_probs = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
    return -np.mean(log_probs[np.arange(batch_size), targets])

def softmax_grad(logits, targets):
    """
    logits: shape (batch_size, seq_len, vocab_size)
    targets: shape (batch_size, seq_len)
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    y_true = np.zeros_like(logits)
    for b in range(batch_size):
        for s in range(seq_len):
            y_true[b, s, targets[b, s]] = 1
    
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    softmax_output = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    return softmax_output - y_true


