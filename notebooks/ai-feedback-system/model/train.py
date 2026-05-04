import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTMClassifier

# same vocab as app.py
vocab = {
    "good": 0,
    "bad": 1,
    "great": 2,
    "terrible": 3,
    "not": 4
}

# simple training data
sentences = [
    ["good"],
    ["bad"],
    ["great"],
    ["terrible"],
    ["not", "good"]
]

labels = [1, 0, 1, 0, 0]

# convert to tensors
X = [torch.tensor([vocab[w] for w in s]) for s in sentences]
X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)

y = torch.tensor(labels).float().unsqueeze(1)

# model
model = LSTMClassifier(len(vocab))

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# train
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ✅ SAVE MODEL (THIS CREATES model.pth)
torch.save(model.state_dict(), "model/model.pth")

print("Model saved to model/model.pth")