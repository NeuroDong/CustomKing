import torch

a = torch.randn(2,2,2,2)
mean = torch.mean(a)
print(mean)