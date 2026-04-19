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

from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor([[1.0,2.0],
                               [2.0,3.0],
                               [3.0,4.0],
                               [5.0,6.0]])
        self.y = torch.tensor([[0.0],[0.0],[1.0],[1.0]])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
from torch.utils.data import DataLoader

dataset = SimpleDataset()

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for X_batch, y_batch in loader:
    print(X_batch, y_batch)
    
for epoch in range(100):

    for X_batch, y_batch in loader:

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
preds = model(X).detach()
pred_labels = (preds > 0.5).float()

accuracy = (pred_labels == y).float().mean()

print("Accuracy:", accuracy.item())

tp = ((pred_labels == 1) & (y == 1)).sum().float()
fp = ((pred_labels == 1) & (y == 0)).sum().float()
fn = ((pred_labels == 0) & (y == 1)).sum().float()

precision = tp / (tp + fp + 1e-8)
recall = tp / (tp + fn + 1e-8)

print("Precision:", precision.item())
print("Recall:", recall.item())

torch.save(model.state_dict(), "model.pth")

model = SimpleNN()  # recreate architecture
model.load_state_dict(torch.load("model.pth"))

model.eval()

with torch.no_grad():
    predictions = model(X)
    print(predictions)