import torch
import torch.nn as nn

vocab_size = 4
embedding_dim = 2

embedding = nn.Embedding(vocab_size, embedding_dim)

word_index = torch.tensor([0])  # "good"

vector = embedding(word_index)
print(vector)

class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.embedding(x).squeeze(1)
        x = torch.sigmoid(self.fc(x))
        return x
        
X = torch.tensor([[0], [1], [2], [3]])
y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

model = TextClassifier()

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