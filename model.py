import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size: int):
        super().d_model=d_model
        self.vocab_size=vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)