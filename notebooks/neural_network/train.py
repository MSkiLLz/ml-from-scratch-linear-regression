import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[1.0, 2.0],
                  [2.0, 3.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])

y = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

model = SimpleNN()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):

    # Forward
    outputs = model(X)

    # Loss
    loss = criterion(outputs, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
print(model(X).detach())