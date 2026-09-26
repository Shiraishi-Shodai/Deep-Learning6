# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, n_head, head_dim, dropout_rate=0.1):
#         super().__init__()
#         self.n_head = n_head
#         self.head_dim = head_dim
#         E, H, D = embed_dim, n_head, head_dim
        
#         # 重みの初期化
#         self.W_q = nn.Linear(E, H*D, bias=False)
#         self.W_k = nn.Linear(E, H*D, bias=False)
#         self.W_v = nn.Linear(E, H*D, bias=False)
#         self.W_o = nn.Linear(H*D, E, bias=False)

#         # ドロップアウトレイヤーの初期化
#         self.attention_dropout = nn.Dropout(dropout_rate)
#         self.output_dropout = nn.Dropout(dropout_rate)
    
#     def forward(self, x):
#         # x : (B, C, E)
#         B, C, E = x.shape
#         H, D = self.n_head, self.head_dim
        
#         Q = self.W_q(x) # (B, C, H*D)
#         K = self.W_k(x) # (B, C, H*D)
#         V = self.W_v(x) # (B, C, H*D)

#         # n_headごとにCとDを持つようデータを整形
#         # (B, C, H*D) → (B, H, C, D)
#         Q = Q.view(B, C, H, D).transpose(1, 2)
#         K = K.view(B, C, H, D).transpose(1, 2)
#         V = V.view(B, C, H, D).transpose(1, 2)

#         # Attentionの重み計算
#         scores = torch.matmul(Q, K.transpose(-1, -2)) # (B, H, C, C)
#         scores = scores / (D ** 0.05)
#         mask = torch.tril(torch.ones((C, C), device=scores.device))
#         scores = scores.masked_fill(mask == 0, float('-inf')) # マスクしたい要素の値をsoftmax後に0にするために-infとする
#         weights = F.softmax(scores, dim=-1)

#         # 出力計算
#         weights = self.attention_dropout(weights)
#         hidden = torch.matmul(weights, V) # (B, H, C, D)

#         # ヘッドの結合
#         hidden = hidden.transpose(1, 2) # (B, C, H, D)
#         hidden = hidden.contiguous().view(B, C, H*D) # (B, C, H*D)

#         # 出力変換
#         output = self.W_o(hidden) # (B, C, E)
#         output = self.output_dropout(output) # (B, C, E)

#         return output

# class LayerNorm(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.ones(embed_dim))
#         self.beta = nn.Parameter(torch.ones(embed_dim))
#         self.eps = 1e-5
    
#     def forward(self, x):
#         mean = x.mean(dim=-1, keepdim=True)
#         var = x.var(dim=-1, keepdim=True)
#         norm_x = (x - mean) / (torch.sqrt(var) + self.eps)
#         return self.gamma * norm_x + self.beta
    
# class GELU(nn.Module):
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(
#             torch.sqrt(torch.tensor(2.0 / torch.pi)) *
#             (x + 0.044715 * torch.pow(x, 3))
#         ))
        
# class FNN(nn.Module):
#     def __init__(self, x_dim, hidden_dim=None, dropout_rate=0.1):
#         super().__init__()
#         if hidden_dim is None:
#             hidden_dim = int(4 * x_dim)
        
#         self.layers = nn.Sequential(
#             nn.Linear(x_dim, hidden_dim),
#             GELU(),
#             nn.Linear(hidden_dim, x_dim),
#             nn.Dropout(dropout_rate)
#         )
    
#     def forward(self, x):
#         return self.layers(x)

# class Block(nn.Module):
#     def __init__(self, embed_dim, n_head, ff_dim=None, dropout_rate=0.1):
#         super().__init__()
#         head_dim = embed_dim // n_head
#         self.attn = MultiHeadAttention(embed_dim=embed_dim, n_head=n_head, head_dim=head_dim, dropout_rate=dropout_rate)
#         self.fnn = FNN(x_dim=embed_dim, hidden_dim=head_dim, dropout_rate=dropout_rate)
#         self.norm1 = LayerNorm(embed_dim=embed_dim)
#         self.norm2 = LayerNorm(embed_dim=embed_dim)
    
#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.fnn(self.norm2(x))
#         return x

# class GPT(nn.Module):
#     def __init__(self, vocab_size, max_context_len, embed_dim, n_head, n_layers, ff_dim, dropout_rate):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.max_context_len = max_context_len
#         self.embed_dim = embed_dim
#         self.n_head = n_head
#         self.n_layers = n_layers
#         self.ff_dim = ff_dim
#         self.dropout_rate = dropout_rate
        
#         # 埋め込み層
#         self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
#         self.pos_embed = nn.Embedding(num_embeddings=max_context_len, embedding_dim=embed_dim)
#         self.dropout = nn.Dropout(dropout_rate)

#         # Transformerブロック
#         self.blocks = nn.ModuleList([
#             Block(embed_dim=embed_dim, n_head=n_head, ff_dim=ff_dim, dropout_rate=dropout_rate)
#             for _ in range(n_layers)
#         ])
        
#         # 出力層
#         self.norm = nn.LayerNorm(embed_dim)
#         self.unembed = nn.Linear(embed_dim, vocab_size)

#         # 重み共有
#         self.embed.weight = self.unembed.weight
        
#         # 重みの初期化
#         self.apply(self.__init__weights)
    
#     def __init__weights(self, module):
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
#     def forward(self, ids):
#         B, C = ids.shape
#         device = ids.device
        
#         # 埋め込み
#         pos = torch.arange(0, C, dtype=torch.long, device=device)
#         emb = self.embed(ids)
#         pos_emb = self.pos_embed(pos)

#         x = self.dropout(emb + pos_emb)

#         # Transformerブロック
#         for block in self.blocks:
#             x = block(x)
        
#         x = self.norm(x)

#         # 出力
#         logits = self.unembed(x) # (B, C, V)
#         return logits
    
#     def save(self, file_path):
#         checkpoint = {
#             "model_state_dict": self.state_dict(),
#             "vocab_size": self.vocab_size,
#             "max_context_len": self.max_context_len,
#             "embed_dim": self.embed_dim,
#             "n_head": self.n_head,
#             "n_layers": self.n_layers,
#             "ff_dim": self.ff_dim,
#             "dropout_rate": self.dropout_rate
#         }
        
#         torch.save(checkpoint)
    
#     @classmethod
#     def load_from(cls, file_path, device="cpu"):
#         checkpoint = torch.load(file_path, map_location=device)

#         model = cls(
#             vocab_size=checkpoint['vocab_size'],
#             max_context_len=checkpoint['max_context_len'],
#             embed_dim=checkpoint['embed_dim'],
#             n_head=checkpoint['n_head'],
#             n_layer=checkpoint['n_layer'],
#             ff_dim=checkpoint['ff_dim'],
#             dropout_rate=checkpoint['dropout_rate']      
#         )
        
#         model.load_state_dict(checkpoint["model_state_dict"])
#         model.to(device)
#         return model

# # B = 2  # バッチサイズ
# # C = 4  # コンテキスト長
# # E = 16 # 埋め込みの次元数
# # H = 3  # ヘッド数
# # D = 8  # 各ヘッドの次元数

# # x = torch.randn(B, C, E)
# # ma = MultiHeadAttention(embed_dim=E, n_head=H, head_dim=D, dropout_rate=0.1)
# # fnn = FNN(x_dim=E, hidden_dim=None, dropout_rate=0.1)
# # output = fnn(x)

# # block = Block(embed_dim=E, n_head=H, ff_dim=None, dropout_rate=0.1)

# vocab_size = 1000
# max_context_len = 256
# embed_dim = 384
# n_head = 6
# n_layer = 6
# ff_dim = 4 * embed_dim
# dropout_rate = 0.1

# model = GPT(
#     vocab_size=vocab_size,
#     max_context_len=max_context_len,
#     embed_dim=embed_dim,
#     n_head=n_head,
#     n_layers=n_layer,
#     ff_dim=ff_dim,
#     dropout_rate=dropout_rate
# )

# dummy_input = torch.randint(0, vocab_size, (3, max_context_len))
# logits = model(dummy_input)
# print(f"入力形状: {dummy_input.shape}")
# print(f"出力形状: {logits.shape}")
# print(model.state_dict())