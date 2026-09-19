import torch
import torch.nn as nn

# val1 = torch.arange(2*3*4).reshape(2, 3, 4)
# val2 = val1.transpose(-2, -1)
# print(val1.shape, val2.shape)

embed = nn.Embedding(10, 5)

x = torch.randint(0, 10, (3, 4))
print(x)

y = embed(x)
print(y.shape)