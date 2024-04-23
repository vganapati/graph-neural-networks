import torch
from torch import nn
import numpy as np

class AttentionLayer(nn.Module):
    def __init__(self, input_len, query_len, value_len):
        super().__init__()
        self.linear_query = nn.Linear(input_len, query_len)
        self.linear_key = nn.Linear(input_len, query_len) # same output dim as query
        self.linear_value = nn.Linear(input_len, value_len)
    def forward(self, x, scale=1, mask=None):
        Q = self.linear_query(x)
        K = self.linear_key(x)
        V = self.linear_value(x)
        output =  scale * (Q @ K.T) # input_len x input_len
        if mask is not None:
            output = mask * output
        softmax = torch.exp(output)/torch.sum(torch.exp(output), axis=1)[None,:]
        return softmax @ V

if __name__ == '__main__':
    num_words = 2
    embedding_len = 4
    query_len = 3
    value_len = 5
    x = torch.randn(num_words, embedding_len)
    attn_layer = AttentionLayer(embedding_len, query_len, value_len)
    mask = torch.ones([num_words, num_words])

    print(attn_layer(x, mask=mask))
