import numpy as np

from train import train
from model.transformer import Transformer
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

if __name__ == "__main__":
    model = test_transformer() 