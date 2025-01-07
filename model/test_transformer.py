import numpy as np

from transformer import Transformer

def test_simple_forward_backward():
    # Model parameters
    d_model = 64
    d_ff = 256
    h = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    vocab_size = 1000
    max_seq_len = 20
    batch_size = 2

    # Initialize model
    model = Transformer(
        d_model=d_model,
        d_ff=d_ff,
        h=h,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len
    )

    # Create dummy data
    src = np.random.randint(0, vocab_size, (batch_size, 10))
    tgt = np.random.randint(0, vocab_size, (batch_size, 11))
    
    try:
        print("Source shape:", src.shape)
        print("Target shape:", tgt[:, :-1].shape)
        print("Embedding shape:", model.embedding.shape)
        
        # Forward pass
        logits = model.forward(src, tgt[:, :-1])
        print("Logits shape:", logits.shape)
        
        # Compute loss
        from utils import cross_entropy_loss
        loss = cross_entropy_loss(logits[:, -1, :], tgt[:, -1])
        print("Loss computed:", loss)
        
        return True
    
    except Exception as e:
        print("Error occurred:", str(e))
        import traceback
        print("Error location:", traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_simple_forward_backward()
    print("\nTest completed successfully!" if success else "\nTest failed!")
