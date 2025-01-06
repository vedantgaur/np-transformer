from model.transformer import Transformer
from model.utils import cross_entropy_loss, softmax_grad

def train(model, train_dataloader, val_dataloader, num_epochs, learning_rate, max_grad_norm=1.0):
    model.learning_rate = learning_rate
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (src, tgt, tgt_y) in enumerate(train_dataloader):
            logits = model.forward(src, tgt)
            loss = cross_entropy_loss(logits, tgt_y)
            
            grad_output = softmax_grad(logits, tgt_y)
            gradients = model.backward(src, tgt, grad_output)
            
            grad_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))
            if grad_norm > max_grad_norm:
                for g in gradients.values():
                    g *= (max_grad_norm / grad_norm)
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        val_loss = validate(model, val_dataloader)
        print(f'Epoch: {epoch}, Training Loss: {total_loss/len(train_dataloader):.4f}, Validation Loss: {val_loss:.4f}')

def validate(model, val_data):
    total_val_loss = 0
    for x_batch, y_batch in val_data:
        predictions = model.forward(x_batch)
        loss = cross_entropy_loss(predictions, y_batch)
        total_val_loss += loss
    return total_val_loss / len(val_data) 

if __name__ == "__main__":
    import numpy as np
    from model.transformer import Transformer

    d_model = 16
    d_ff = 64
    h = 4
    num_layers = 2
    num_epochs = 10
    learning_rate = 0.001

    train_data = [(np.random.randn(8, 10, d_model), np.random.randint(0, d_model, 8)) for _ in range(100)]
    val_data = [(np.random.randn(8, 10, d_model), np.random.randint(0, d_model, 8)) for _ in range(20)]

    model = Transformer(d_model, d_ff, h, num_layers)

    train(model, train_data, val_data, num_epochs, learning_rate)
