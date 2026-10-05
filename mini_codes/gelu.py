import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# GELU
gelu_activation = nn.GELU()

# 入力
input_data = torch.arange(
    -5.0, 5.0, 0.1,
    requires_grad=True
)

# Forward
output = gelu_activation(input_data)

# Backward
loss = output.sum()
loss.backward()

print(f"勾配の最小値: {min(input_data.grad)}, index: {torch.argmin(input_data.grad)}")
print(f"勾配が最小となった入力値: {input_data[torch.argmin(input_data.grad).item()]}")
# 勾配の最小値: -0.12886172533035278, index: 36
# 勾配が最小となった入力値: -1.399999976158142

print(f"勾配の最大値: {max(input_data.grad)}, index: {torch.argmax(input_data.grad)}")
print(f"勾配が最大となった入力値: {input_data[torch.argmax(input_data.grad).item()]}")
# 勾配の最大値: 1.128861665725708, index: 64
# 勾配が最大となった入力値: 1.399999976158142


# グラフ
fig, ax = plt.subplots(figsize=(10, 6))

# Forward
ax.plot(
    input_data.detach(),
    output.detach(),
    label="Forward: GELU(x)"
)

# Backward
ax.plot(
    input_data.detach(),
    input_data.grad,
    label="Backward: dGELU/dx"
)

# x=0, y=0 の補助線
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)

ax.set_xlabel("Input (x)")
ax.set_ylabel("Value")
ax.set_title("GELU Forward and Backward")
ax.legend()
ax.grid()

plt.savefig("imgs/GELU.png")