import numpy as np
from model.decoder import decoder, decoder_backward
from model.encoder import encoder, encoder_backward


class Transformer:
    def __init__(self, d_model, d_ff, h, num_encoder_layers, num_decoder_layers, vocab_size, max_seq_len):
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.embedding = np.random.normal(
            0, np.sqrt(2.0 / (vocab_size + d_model)), 
            (vocab_size, d_model)
        )
        
        self.positional_encoding = self.generate_positional_encoding(max_seq_len, d_model)

    def generate_positional_encoding(self, max_seq_len, d_model):
        pos = np.arange(max_seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)

        pos_encoding = pos * angle_rates
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
        return pos_encoding
    
    def forward(self, src, tgt):
        src_embedded = src @ self.embedding.T
        tgt_embedded = tgt @ self.embedding.T
        
        src_pos = src_embedded + self.positional_encoding[:src.shape[1]]
        tgt_pos = tgt_embedded + self.positional_encoding[:tgt.shape[1]]
        
        enc_output = src_pos
        for _ in range(self.num_encoder_layers):
            enc_output = encoder(enc_output, self.d_ff, self.d_model, self.h)
        
        dec_output = tgt_pos
        for _ in range(self.num_decoder_layers):
            dec_output = decoder(dec_output, enc_output, self.d_ff, self.d_model, self.h)
        
        logits = dec_output @ self.embedding
        return logits
    
    def backward(self, src, tgt, grad_output):
        grad_embedding = grad_output
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

        self.embedding -= self.learning_rate * (
            grad_src_embed.T @ src + 
            grad_tgt_embed.T @ tgt + 
            grad_embedding
        )

        return {
            'grad_embedding': grad_embedding,
            'grad_src_embed': grad_src_embed,
            'grad_tgt_embed': grad_tgt_embed
        }
    

