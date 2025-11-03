import math
from torch import nn
import torch
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, d_embedding=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_embedding = d_embedding
        self.d_k = d_embedding // heads
        self.h = heads
        self.input_projection = nn.Linear(d_model, d_embedding)
        self.q_linear = nn.Linear(d_embedding, d_embedding)
        self.v_linear = nn.Linear(d_embedding, d_embedding)
        self.k_linear = nn.Linear(d_embedding, d_embedding)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_embedding, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q_proj = self.input_projection(q)
        k_proj = self.input_projection(k)
        v_proj = self.input_projection(v)
        k = self.k_linear(k_proj).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q_proj).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v_proj).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_embedding)
        output = self.out(concat)
        return output