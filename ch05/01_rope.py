import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, theta, key_dim, max_context_len):
        super().__init__()
        assert key_dim %2 == 0 # 2次元平面上の角度を使うため、key_dimは偶数とする
        

rope = RoPE(1, 4, 10)
