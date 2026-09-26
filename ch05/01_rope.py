import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, theta, key_dim, max_context_len):
        super().__init__()
        assert key_dim %2 == 0 # 2次元平面上の角度を使うため、key_dimは偶数とする
        half = key_dim // 2
        
        half_ids = torch.arange(0, half)
        inv_freq = 1.0 / (theta ** ((2.0 * half_ids) / key_dim)) # (half, )

        positions = torch.arange(max_context_len) # (max_context_len)
        angles = positions[:, None] ** inv_freq[None, :] # (max_context_len, half)
        
        cos = torch.cos(angles) # (max_context_len, half)
        sin = torch.sin(angles) # (max_context_len, half)

rope = RoPE(1, 4, 10)
print(torch.arange(10)[:, None].shape)