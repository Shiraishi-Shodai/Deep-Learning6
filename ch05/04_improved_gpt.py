import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, theta, key_dim, max_context_len):
        super().__init__()
        assert key_dim % 2 == 0
        half = key_dim // 2
        
        half_ids = torch.arange(0, half)
        inv_freq = 1.0 / (theta ** ((2.0 * half_ids) / key_dim))
        
        positions = torch.arange(max_context_len)
        angles = positions[:, None] * inv_freq[None, *]

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos_cache", cos)
        self.register_buffer("sin_cache", sin)
    
    def forward(self, x):
        batch_size, num_head, context_len, key_dim = x.shape