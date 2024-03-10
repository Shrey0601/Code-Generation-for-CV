import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

embedding_dim = 300  # size of each embedding vector
vocab_size = len(model.wv.vocab)  # size of the vocabulary
weights = torch.FloatTensor(model.wv.vectors)  # the embedding weights from Word2Vec

# create the embedding layer
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
embedding_layer.load_state_dict({'weight': weights})

class SparseAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_prob, feedforward_dim=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        
        self.head_dim = hidden_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.qkv_projection = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout_prob)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 2
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        qkv = self.qkv_projection(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim).transpose(2, 3)
        q, k, v = qkv[...,:self.head_dim], qkv[...,self.head_dim:2*self.head_dim], qkv[...,2*self.head_dim:]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        x = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        
        x = self.output_projection(x)
        x = self.feedforward(x)
        return x

    
class SparseTransformer(nn.Module):
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
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)  # replace nn.Linear with nn.Embedding
        self.input_embedding.load_state_dict({'weight': weights})  # load the pre-trained embeddings
        self.output_embedding = nn.Linear(hidden_dim, output_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, feedforward_dim, dropout_prob)
            for _ in range(num_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            SparseAttentionDecoderLayer(hidden_dim, num_heads, feedforward_dim, dropout_prob)  # using SparseAttention
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


data = pd.read_csv("dataset.csv")
vocab = []
for line in data["Task"]:
    for word in line.split():
        if word not in vocab:
            vocab.append(word)

print(len(vocab))