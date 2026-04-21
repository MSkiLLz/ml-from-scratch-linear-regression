import torch

texts = ["good", "bad", "great", "terrible"]
labels = [1, 0, 1, 0]

vocab = {"good": 0, "bad": 1, "great": 2, "terrible": 3}

X = [vocab[word] for word in texts]
X = torch.tensor(X).float().unsqueeze(1)

y = torch.tensor(labels).float().unsqueeze(1)

import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = TextClassifier()

import torch.optim as optim

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(loss.item())
        
        
test_word = "good"
test_input = torch.tensor([[vocab[test_word]]]).float()

with torch.no_grad():
    pred = model(test_input)

print("Prediction:", pred.item())        