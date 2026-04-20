import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch

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

dataset = SimpleDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

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

    for X_batch, y_batch in loader:

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
with torch.no_grad():
    preds = model(dataset.X)
    pred_labels = (preds > 0.5).float()

accuracy = (pred_labels == dataset.y).float().mean()
print("Accuracy:", accuracy.item())

torch.save(model.state_dict(), "model.pth")

model.eval()

new_data = torch.tensor([[4.0, 5.0]])

with torch.no_grad():
    prediction = model(new_data)

print("Prediction:", prediction.item())