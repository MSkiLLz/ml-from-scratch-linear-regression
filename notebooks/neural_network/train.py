import torch

Q = torch.rand(3, 4)
K = torch.rand(3, 4)
V = torch.rand(3, 4)

scores = torch.matmul(Q, K.T)
weights = torch.softmax(scores, dim=1)

output = torch.matmul(weights, V)