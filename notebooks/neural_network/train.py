import torch
import torch.nn as nn

sentences = [
    ["good", "movie"],
    ["not", "good"],
    ["great", "film"],
    ["bad", "movie"]
]

labels = [1, 0, 1, 0]

vocab = {"good":0, "movie":1, "not":2, "great":3, "film":4, "bad":5}

X = [[vocab[word] for word in sentence] for sentence in sentences]

X = torch.tensor(X)
y = torch.tensor(labels).float().unsqueeze(1)

embedding = nn.Embedding(len(vocab), 4)

embedded = embedding(X)
print(embedded.shape)

avg_embedding = embedded.mean(dim=1)

class SimpleTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), 4)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = torch.sigmoid(self.fc(x))
        return x
        

model = SimpleTextModel()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(loss.item())

print(model.embedding.weight)