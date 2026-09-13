import torch

val1 = torch.arange(2*3*4).reshape(2, 3, 4)
val2 = val1.transpose(-2, -1)
print(val1.shape, val2.shape)