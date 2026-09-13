import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim, key_dim):
        super().__init__()

        self.W_q = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_o = nn.Linear(key_dim, embed_dim, bias=False)
        self.key_dim = key_dim
    
    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        K_t = K.transpose(-1, -2)
        scores = torch.matmul(Q, K_t)
        scores = scores / (self.key_dim ** 0.5)

        B, C, E = x.shape
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        hidden = torch.matmul(weights, V)
        
        output = self.W_o(hidden)

        return output

embed_dim = 256
key_dim = 64
attention = Attention(embed_dim, key_dim)
x = torch.randn(2, 5, embed_dim)
y = attention(x)

print(f"入力形状: {x.shape}")
print(f"出力形状: {y.shape}")
    