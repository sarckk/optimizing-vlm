from torch import nn

from .attention import MultiHeadAttention

class Block(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, num_heads, dropout, is_decoder)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        original_x = x  # Save for residual connection
        x = self.ln1(x)
        attn_output = self.attn(x)
        x = original_x + attn_output
        x = self.ln2(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        return x