import torch
from matplotlib import pyplot as plt

input_data = torch.arange(-8.0, 8, 0.1, requires_grad=True)
output = torch.relu(input_data)
loss = output.sum()
loss.backward()

# グラフ
fig, ax = plt.subplots(figsize=(10, 6))

# Forward
ax.plot(
    input_data.detach(),
    output.detach(),
    label="Forward: ReLU(x)"
)

# Backward
ax.plot(
    input_data.detach(),
    input_data.grad,
    label="Backward: dReLU/dx"
)

# x=0, y=0 の補助線
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)

ax.set_xlabel("Input (x)")
ax.set_ylabel("Value")
ax.set_title("ReLU Forward and Backward")
ax.legend()
ax.grid()

plt.savefig("imgs/ReLU.png")