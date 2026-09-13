import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim
        
        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)
        
        # Dropoutを追加
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim
        
        Q = self.W_q(x) # (B, C, H*D)
        K = self.W_k(x) # (B, C, H*D)
        V = self.W_v(x) # (B, C, H*D)
        
        Q = Q.view(B, C, H, D).transpose(1, 2) # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2) # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2) # (B, H, C, D)
        
        # QKとスケール処理
        scores = torch.matmul(Q, K.transpose(-1, -2)) # (B, H, C, C)
        scores = scores / (D ** 0.5)
        
        # マスク処理
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # attentionの重み計算
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        hidden = torch.matmul(weights, V) # (B, H, C, D)
        
        # ヘッドの結合
        hidden = hidden.transpose(1, 2) # (B, C, H, D)
        hidden = hidden.contiguous().view(B, C, H*D) # (B, C, H*D)
        
        # 出力変換
        output = self.W_o(hidden) # (B, C, E)
        output = self.output_dropout(output)
        
        return output

class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.ones(embed_dim))
        self.eps = 1e-5
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta
    

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        
class FNN(nn.Module):
    def __init__(self, x_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * x_dim)
        
        self.layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, x_dim),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        return self.layers(x)

class Block(nn.Module):
    def __init__(self, embed_dim, n_head, ff_dim=None, dropout_rate=0.1):
        super().__init__()
        head_dim = embed_dim // n_head
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_head, head_dim, dropout_rate)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FNN(embed_dim, ff_dim, dropout_rate)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
