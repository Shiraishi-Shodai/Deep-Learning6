import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x = torch.tensor([100.0, 200.0, 300.0])
y = F.softmax(x, dim=0)

d = 10
q = np.random.randn(d)
k = np.random.randn(d)

dot_product = np.dot(q, k)
scaled_dot_product = dot_product / np.sqrt(d)

print(f"dot product : {dot_product}")
print(f"scaled dot product: {scaled_dot_product}")

num_samples = 10000
dot_products = []
scaled_dot_products = []

for _ in range(num_samples):
    k = np.random.randn(d)
    q = np.random.randn(d)
    dot_product = np.dot(q, k)
    scaled_dot_product = dot_product / np.sqrt(d)
    
    dot_products.append(dot_product)
    scaled_dot_products.append(scaled_dot_product)

# 結果をプロット
plt.figure(figsize=(10, 6))
plt.hist(dot_products, bins=50, alpha=0.5, label='Without scaling')
plt.hist(scaled_dot_products, bins=50, alpha=0.5, label='With scaling')
plt.legend()
plt.show()

print("Variances without scaling:", np.var(dot_products))
print("Variances with scaling:", np.var(scaled_dot_products))
