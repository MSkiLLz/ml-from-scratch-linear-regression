import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x)

X = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
                  
w = torch.tensor([0.5, 0.2])

z = torch.matmul(X, w)
print(z)

x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x

y.backward()

print(x.grad)

