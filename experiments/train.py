import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model.transformer import Transformer
from model.utils import cross_entropy_loss, softmax_grad
from data.dataset import TransformerDataset

def train(model, train_dataset, val_dataset, num_epochs, learning_rate, save_dir='checkpoints'):
    print("\nStarting training...")
    print(f"Number of epochs: {num_epochs}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Total batches per epoch: {len(train_dataset)}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    warmup_steps = 4000
    min_lr = learning_rate / 100
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        total_loss = 0
        num_batches = len(train_dataset)
        
        for i in range(num_batches):
            step = epoch * num_batches + i
            if step < warmup_steps:
                current_lr = learning_rate * (step / warmup_steps)
            else:
                current_lr = max(min_lr, learning_rate * 0.99 ** (step - warmup_steps))
            
            model.learning_rate = current_lr
            
            src, tgt_input, tgt_y = train_dataset[i]
            
            logits = model.forward(src, tgt_input)
            loss = cross_entropy_loss(logits.reshape(-1, logits.shape[-1]), tgt_y.reshape(-1))
            
            if i % 10 == 0:
                print(f"\nBatch {i+1}/{num_batches}")
                print(f"Loss: {loss:.4f}")
                print(f"Learning rate: {current_lr:.6f}")
            
            grad_output = softmax_grad(logits, tgt_y)
            model.backward(src, tgt_input, grad_output)
            
            total_loss += loss
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1} - Average loss: {avg_loss:.4f}")
        
        # Validation
        val_loss = validate(model, val_dataset)
        print(f"Validation loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model(f"{save_dir}/best_model.npy")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break

def validate(model, val_dataloader):
    model.eval()
    total_val_loss = 0
    for batch_idx in range(len(val_dataloader)):
        src, tgt, tgt_y = val_dataloader[batch_idx]
        logits = model.forward(src, tgt)
        loss = cross_entropy_loss(logits.reshape(-1, logits.shape[-1]), tgt_y.reshape(-1))
        total_val_loss += loss
    return total_val_loss / len(val_dataloader)

if __name__ == "__main__":
    d_model = 512
    d_ff = 2048
    h = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    vocab_size = 32000
    max_seq_len = 512
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001

    num_train_samples = 1000
    num_val_samples = 200
    seq_len = 20

    src_train = np.random.randint(0, vocab_size, (num_train_samples, seq_len))
    tgt_train = np.random.randint(0, vocab_size, (num_train_samples, seq_len + 1))
    
    src_val = np.random.randint(0, vocab_size, (num_val_samples, seq_len))
    tgt_val = np.random.randint(0, vocab_size, (num_val_samples, seq_len + 1))

    train_dataset = TransformerDataset(src_train, tgt_train, batch_size=batch_size)
    val_dataset = TransformerDataset(src_val, tgt_val, batch_size=batch_size)

    model = Transformer(
        d_model=d_model,
        d_ff=d_ff,
        h=h,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len
    )

    train(model, train_dataset, val_dataset, num_epochs, learning_rate)
