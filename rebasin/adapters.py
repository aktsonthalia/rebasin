import torch
from torch import nn

from einops import rearrange    

class AttentionForRebasin(nn.Module):

    def __init__(self, dims, heads, scale):
        
        super(AttentionForRebasin, self).__init__()

        self.dims = dims
        self.heads = heads
        self.scale = scale

        self.query_encoder = nn.Linear(dims, dims, bias=False)
        self.query_encoder.tag = 'query_encoder'
        self.key_encoder = nn.Linear(dims, dims, bias=False)
        self.key_encoder.tag = 'key_encoder'
        self.value_encoder = nn.Linear(dims, dims, bias=False)
        self.value_encoder.tag = 'value_encoder'
        
        self.attend = nn.Softmax(dim = -1)

    def forward(self, x):

        q = self.query_encoder(x)
        k = self.key_encoder(x)
        v = self.value_encoder(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return out
