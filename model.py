import math

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, nhead, nhid, nlayers):
        super(MaskedLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoder = PositionalEncoding(embedding_size)
        encoder_layers = TransformerEncoderLayer(embedding_size, nhead, nhid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
