import torch
import torch.nn as nn
from matplotlib import pyplot as plt

"""RMSNorm後は、二乗平均が1になるように整形される。ただ、平均で引く処理をしていないため、-1次元の平均は0にならない。
"""

class RMSNorm(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(x))
        self.eps = 1e-5
        
    def forward(self, x):
        x2 = x**2
        ms = x2.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        return self.gamma * x / rms
    

# -------------------------
# 入力データ
# -------------------------
torch.manual_seed(0)

# 100個のベクトル
# 1つのベクトルは10次元
x = torch.randn(100, 10)

# ベクトルごとに異なるスケールを持たせる
x[0:30] *= 0.1
x[30:60] *= 1
x[60:100] *= 100

# -------------------------
# RMSNorm
# -------------------------
rmsnorm = RMSNorm(10)
out = rmsnorm(x)

# -------------------------
# RMSを計算
# -------------------------
rms_before = torch.sqrt(
    (x ** 2).mean(dim=-1)
)

rms_after = torch.sqrt(
    (out ** 2).mean(dim=-1)
)

# -------------------------
# RMSの比較
# -------------------------
# plt.figure(figsize=(10, 5))

# plt.plot(rms_before.detach().numpy(), label="Before RMSNorm")
# plt.plot(rms_after.detach().numpy(), label="After RMSNorm")

# plt.xlabel("Vector")
# plt.ylabel("RMS")

# plt.legend()
# plt.grid()

# plt.show()

# 代表的な3つのベクトルを表示(二乗平均)
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

indices = [10, 45, 80]

for ax, idx in zip(axes, indices):

    ax.plot(
        x[idx].detach().numpy(),
        marker="o",
        label="Before"
    )

    ax.plot(
        out[idx].detach().numpy(),
        marker="o",
        label="After RMSNorm"
    )

    ax.set_title(
        f"Vector {idx} "
        f"(RMS before={rms_before[idx]:.2f}, "
        f"after={rms_after[idx]:.2f})"
    )

    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()

# 平均を表示
print("RMS before:", rms_before[:5])
print("RMS after :", rms_after[:5])

print("Mean before:", x.mean(dim=-1)[:5])
print("Mean after :", out.mean(dim=-1)[:5])