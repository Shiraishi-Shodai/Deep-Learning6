import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_head, head_dim, dropout_rate=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim
        
        # 重みの初期化
        self.W_q = nn.Linear(E, H*D, bias=False)
        self.W_k = nn.Linear(E, H*D, bias=False)
        self.W_v = nn.Linear(E, H*D, bias=False)
        self.W_o = nn.Linear(H*D, E, bias=False)

        # ドロップアウトレイヤーの初期化
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x : (B, C, E)
        B, C, E = x.shape
        H, D = self.n_head, self.head_dim
        
        Q = self.W_q(x) # (B, C, H*D)
        K = self.W_k(x) # (B, C, H*D)
        V = self.W_v(x) # (B, C, H*D)

        # n_headごとにCとDを持つようデータを整形
        # (B, C, H*D) → (B, H, C, D)
        Q = Q.view(B, C, H, D).transpose(1, 2)
        K = K.view(B, C, H, D).transpose(1, 2)
        V = V.view(B, C, H, D).transpose(1, 2)

        # Attentionの重み計算
        scores = torch.matmul(Q, K.transpose(-1, -2)) # (B, H, C, C)
        scores = scores / (D ** 0.05)
        mask = torch.tril(torch.ones((C, C), device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf')) # マスクしたい要素の値をsoftmax後に0にするために-infとする
        weights = F.softmax(scores, dim=-1)

        # 出力計算
        weights = self.attention_dropout(weights)
        hidden = torch.matmul(weights, V) # (B, H, C, D)

        # ヘッドの結合
        hidden = hidden.transpose(1, 2) # (B, C, H, D)
        hidden = hidden.contiguous().view(B, C, H*D) # (B, C, H*D)

        # 出力変換
        output = self.W_o(hidden) # (B, C, E)
        output = self.output_dropout(output) # (B, C, E)

        return output

class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.ones(embed_dim))
        self.eps = 1e-5
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / (torch.sqrt(var) + self.eps)
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
        self.attn = MultiHeadAttention(embed_dim=embed_dim, n_head=n_head, head_dim=head_dim, dropout_rate=dropout_rate)
        self.fnn = FNN(x_dim=embed_dim, hidden_dim=head_dim, dropout_rate=dropout_rate)
        self.norm1 = LayerNorm(embed_dim=embed_dim)
        self.norm2 = LayerNorm(embed_dim=embed_dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.fnn(self.norm2(x))
        return x
    
B = 2  # バッチサイズ
C = 4  # コンテキスト長
E = 16 # 埋め込みの次元数
H = 3  # ヘッド数
D = 8  # 各ヘッドの次元数

x = torch.randn(B, C, E)
# ma = MultiHeadAttention(embed_dim=E, n_head=H, head_dim=D, dropout_rate=0.1)
# fnn = FNN(x_dim=E, hidden_dim=None, dropout_rate=0.1)
# output = fnn(x)

block = Block(embed_dim=E, n_head=H, ff_dim=None, dropout_rate=0.1)
output = block(x)
print(output.shape)