import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model.transformer import Transformer
from model.utils import cross_entropy_loss, softmax_grad
from data.dataset import TransformerDataset

def train(model, train_dataset, val_dataset, num_epochs, learning_rate):
    print("\nStarting training...")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    
    model.learning_rate = learning_rate  
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        total_loss = 0
        num_batches = 0
        
        for i, (src, tgt_input, tgt_y) in enumerate(train_dataset):
            print(f"\nBatch {i+1}")
            print(f"Source shape: {src.shape}")
            print(f"Target input shape: {tgt_input.shape}")
            print(f"Target output shape: {tgt_y.shape}")
            
            # Forward pass
            logits = model.forward(src, tgt_input)
            print(f"Logits shape: {logits.shape}")
            
            # Compute loss
            loss = cross_entropy_loss(logits.reshape(-1, logits.shape[-1]), tgt_y.reshape(-1))
            print(f"Batch loss: {loss:.4f}")
            
            # Backward pass
            grad_output = softmax_grad(logits, tgt_y)
            print(f"Gradient output shape: {grad_output.shape}")
            
            gradients = model.backward(src, tgt_input, grad_output)
            print("Gradient shapes:")
            for key, grad in gradients.items():
                print(f"  {key}: {grad.shape}")
            
            total_loss += loss
            num_batches += 1
            
            if i % 10 == 0:
                print(f"Batch {i+1}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1} - Average loss: {avg_loss:.4f}")
        
        # Validation
        val_loss = validate(model, val_dataset)
        print(f"Validation loss: {val_loss:.4f}")

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
