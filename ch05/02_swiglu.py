import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, x_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(x_dim * 8 / 3)
        
        self.W = nn.Linear(x_dim, hidden_dim, bias=False)
        self.V = nn.Linear(x_dim, hidden_dim, bias=False)
        self.O = nn.Linear(hidden_dim, x_dim, bias=False)
    
    def forward(self, x):
        a = self.W(x)
        b = self.V(x)

        gated = F.silu(a) * b
        out = self.O(gated)
        return out

x = torch.arange(-6, 3, 0.1).reshape(-1, 1)
# out = silu(x)
swiglu = SwiGLU(x.shape[-1], hidden_dim=None)
out = swiglu(x)

plt.plot(x.detach().numpy(), out.detach().numpy())
plt.grid()
plt.show()

