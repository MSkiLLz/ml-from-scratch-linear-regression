import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x
        
model = SimpleNN()
print(model)

X = torch.tensor([[1.0, 2.0],
                  [2.0, 3.0]])

output = model(X)

print(output)

for name, param in model.named_parameters():
    print(name, param.shape)