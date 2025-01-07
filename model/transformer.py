import numpy as np
from model.decoder import decoder, decoder_backward
from model.encoder import encoder, encoder_backward


class Transformer:
    def __init__(self, d_model, d_ff, h, num_encoder_layers, num_decoder_layers, vocab_size, max_seq_len, learning_rate=0.0001):
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate

        self.embedding = np.random.normal(
            0, np.sqrt(2.0 / (vocab_size + d_model)), 
            (vocab_size, d_model)
        )
        
        self.positional_encoding = self.generate_positional_encoding(max_seq_len, d_model)

        self.encoder_layers = [{
            'W_Q': np.random.normal(0, 0.02, (d_model, d_model)),
            'W_K': np.random.normal(0, 0.02, (d_model, d_model)),
            'W_V': np.random.normal(0, 0.02, (d_model, d_model)),
            'W_O': np.random.normal(0, 0.02, (d_model, d_model)),
            'W_1': np.random.normal(0, 0.02, (d_model, d_ff)),
            'W_2': np.random.normal(0, 0.02, (d_ff, d_model)),
            'b_1': np.zeros(d_ff),
            'b_2': np.zeros(d_model),
        } for _ in range(num_encoder_layers)]

        self.decoder_layers = [{
            'self_W_Q': np.random.normal(0, 0.02, (d_model, d_model)),
            'self_W_K': np.random.normal(0, 0.02, (d_model, d_model)),
            'self_W_V': np.random.normal(0, 0.02, (d_model, d_model)),
            'cross_W_Q': np.random.normal(0, 0.02, (d_model, d_model)),
            'cross_W_K': np.random.normal(0, 0.02, (d_model, d_model)),
            'cross_W_V': np.random.normal(0, 0.02, (d_model, d_model)),
            'W_O': np.random.normal(0, 0.02, (d_model, d_model)),
            'W_1': np.random.normal(0, 0.02, (d_model, d_ff)),
            'W_2': np.random.normal(0, 0.02, (d_ff, d_model)),
            'b_1': np.zeros(d_ff),
            'b_2': np.zeros(d_model),
        } for _ in range(num_decoder_layers)]

        self.training = True

    def generate_positional_encoding(self, max_seq_len, d_model):
        pos = np.arange(max_seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)

        pos_encoding = pos * angle_rates
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
        return pos_encoding
    
    def forward(self, src, tgt):
        assert len(src.shape) == 2, f"Expected src shape (batch_size, seq_len), got {src.shape}"
        assert len(tgt.shape) == 2, f"Expected tgt shape (batch_size, seq_len), got {tgt.shape}"
        
        src_embedded = self.embedding[src]
        tgt_embedded = self.embedding[tgt]
        
        src_pos = src_embedded + self.positional_encoding[:src.shape[1]]
        tgt_pos = tgt_embedded + self.positional_encoding[:tgt.shape[1]]
        
        enc_output = src_pos
        for _ in range(self.num_encoder_layers):
            enc_output = encoder(enc_output, self.d_ff, self.d_model, self.h)
        
        dec_output = tgt_pos
        for _ in range(self.num_decoder_layers):
            dec_output = decoder(dec_output, enc_output, self.d_ff, self.d_model, self.h)
        
        assert dec_output.shape[-1] == self.d_model, \
            f"Expected decoder output dim {self.d_model}, got {dec_output.shape[-1]}"
        
        logits = dec_output @ self.embedding.T
        
        assert logits.shape == (src.shape[0], tgt.shape[1], self.vocab_size), \
            f"Expected logits shape {(src.shape[0], tgt.shape[1], self.vocab_size)}, got {logits.shape}"
        
        return logits
    
    def backward(self, src, tgt, grad_output):
        grad_embedding = grad_output @ np.eye(self.vocab_size)
        grad_dec_output = grad_output @ self.embedding

        grad_dec = grad_dec_output
        grad_enc_outputs = []
        for _ in range(self.num_decoder_layers):
            grad_dec, grad_enc = decoder_backward(
                grad_dec,
                self.d_ff,
                self.d_model,
                self.h
            )
            grad_enc_outputs.append(grad_enc)

        grad_enc = sum(grad_enc_outputs)
        
        for _ in range(self.num_encoder_layers):
            grad_enc = encoder_backward(
                grad_enc,
                self.d_ff,
                self.d_model,
                self.h
            )

        grad_src_embed = grad_enc
        grad_tgt_embed = grad_dec

        src_one_hot = np.eye(self.vocab_size)[src]
        tgt_one_hot = np.eye(self.vocab_size)[tgt]
        
        grad_src = grad_src_embed.reshape(-1, self.d_model).T @ src_one_hot.reshape(-1, self.vocab_size)
        grad_tgt = grad_tgt_embed.reshape(-1, self.d_model).T @ tgt_one_hot.reshape(-1, self.vocab_size)
        
        batch_size, seq_len, _ = grad_output.shape
        full_grad_embedding = np.zeros((self.vocab_size, self.d_model))
        
        token_grads = grad_output.reshape(-1, self.vocab_size)
        
        token_indices = token_grads.argmax(axis=1)
        
        for idx, token_idx in enumerate(token_indices):
            full_grad_embedding[token_idx] += token_grads[idx, token_idx] * self.embedding[token_idx]
        
        grad_src = grad_src.T
        grad_tgt = grad_tgt.T
        
        self.embedding -= self.learning_rate * (
            grad_src + grad_tgt + full_grad_embedding
        )

        return {
            'grad_embedding': full_grad_embedding,
            'grad_src_embed': grad_src_embed,
            'grad_tgt_embed': grad_tgt_embed
        }
    
    def train(self):
        """Set the model to training mode"""
        self.training = True
    
    def eval(self):
        """Set the model to evaluation mode"""
        self.training = False
    

