import torch
from torch import nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, nhead, nhid, nlayers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.transformer = nn.Transformer(embedding_size, nhead, nlayers, nhid)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src)
        output = self.transformer(embedded)
        output = self.fc(output)
        return output