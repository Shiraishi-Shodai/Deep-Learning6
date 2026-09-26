import torch

x = torch.arange(2*3*4).reshape(2, 3, 4)
# 偶数・奇数インデックスに分割
x_even = x[..., 0::2]
x_odd = x[..., 1::2]

print(x)
print(x_even)
print(x_odd)
print()
x_stack = torch.stack([x_even, x_odd], dim=-1)
out = x_stack.reshape(2, 3, 4)

print(x_stack)
print(x_stack.shape)
print()
print(out)