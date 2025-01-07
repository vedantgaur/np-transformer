import numpy as np

from train import train
from model.transformer import Transformer
from model.utils import cross_entropy_loss
from data.dataset import TransformerDataset

def test_transformer():
    d_model = 512
    d_ff = 2048
    h = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    vocab_size = 32000
    max_seq_len = 512
    batch_size = 32
    
    model = Transformer(
        d_model=d_model,
        d_ff=d_ff,
        h=h,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len
    )
    
    src = np.random.randint(0, vocab_size, (batch_size, 10))
    tgt = np.random.randint(0, vocab_size, (batch_size, 11))
    

    train_dataset = TransformerDataset(src, tgt, batch_size=batch_size)
    val_dataset = TransformerDataset(src, tgt, batch_size=batch_size)
    

    train(
        model=model,
        train_dataloader=train_dataset,
        val_dataloader=val_dataset,
        num_epochs=10,
        learning_rate=0.0001
    )
    
    return model

def test_trained_model(model, test_dataset):
    print("\nTesting trained model...")
    model.eval()
    
    total_loss = 0
    num_correct = 0
    total_tokens = 0
    
    for i, (src, tgt_input, tgt_y) in enumerate(test_dataset):
        # Forward pass
        logits = model.forward(src, tgt_input)
        
        # Compute loss
        loss = cross_entropy_loss(logits.reshape(-1, logits.shape[-1]), tgt_y.reshape(-1))
        total_loss += loss
        
        # Compute accuracy
        predictions = logits.argmax(axis=-1)
        correct = (predictions == tgt_y).sum()
        num_correct += correct
        total_tokens += tgt_y.size
        
        if i % 5 == 0:
            print(f"\nBatch {i+1}")
            print(f"Loss: {loss:.4f}")
            print(f"Batch accuracy: {correct/tgt_y.size:.2%}")
            
            # Show some example predictions
            for j in range(min(2, len(src))):
                print(f"\nExample {j+1}:")
                print(f"Target:     {tgt_y[j]}")
                print(f"Prediction: {predictions[j]}")
    
    avg_loss = total_loss / len(test_dataset)
    accuracy = num_correct / total_tokens
    
    print("\nTest Results:")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return avg_loss, accuracy

if __name__ == "__main__":
    # Training
    model = test_transformer()
    
    # Save the trained model
    model.save_model('trained_model.npy')
    
    # Load the model for testing
    test_model = Transformer.from_pretrained('trained_model.npy')
    
    # Create test dataset
    num_test_samples = 100
    seq_len = 20
    vocab_size = 32000
    
    src_test = np.random.randint(0, vocab_size, (num_test_samples, seq_len))
    tgt_test = np.random.randint(0, vocab_size, (num_test_samples, seq_len + 1))
    
    test_dataset = TransformerDataset(src_test, tgt_test, batch_size=32)
    
    # Test the loaded model
    test_loss, test_accuracy = test_trained_model(test_model, test_dataset) 