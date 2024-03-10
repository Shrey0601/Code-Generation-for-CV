import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, feedforward_dim, dropout_prob):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout_prob = dropout_prob
        
        # Embedding layers
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        self.output_embedding = nn.Linear(hidden_dim, output_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, feedforward_dim, dropout_prob)
            for _ in range(num_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(hidden_dim, num_heads, feedforward_dim, dropout_prob)
            for _ in range(num_layers)
        ])
        
        # Output projection layer
        self.output_projection = nn.Linear(output_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, src, tgt):
        # Embed the inputs
        src = self.input_embedding(src)
        tgt = self.input_embedding(tgt)
        
        # Encode the source sequence
        for layer in self.encoder_layers:
            src = layer(src)
        
        # Decode the target sequence
        for layer in self.decoder_layers:
            tgt = layer(tgt, src)
        
        # Project the output sequence
        tgt = self.output_projection(tgt)
        tgt = F.relu(tgt)
        tgt = self.dropout(tgt)
        tgt = self.output_embedding(tgt)
        
        return tgt
