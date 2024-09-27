import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()
        assert n_embd % num_heads == 0, "n_embd must be divisible by num_heads"
        self.wqkv = nn.Linear(n_embd, n_embd * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout_pct = dropout
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        self.is_decoder = is_decoder

    def forward(self, x):
        B, T, C = x.shape
        assert C == self.n_embd
        q, k, v = self.wqkv(x).split([self.n_embd, self.n_embd, self.n_embd], dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        mask = None
        if self.is_decoder:
            # Upper triangular mask
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)) != 0

        # Uses flash attention by default
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout_pct
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_embd)

        out = self.dropout(self.proj(out))
        return out
