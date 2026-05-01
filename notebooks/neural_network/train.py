import torch
import torch.nn as nn

num_users = 3
num_items = 3
embedding_dim = 4

class RecSysModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)

        x = torch.cat([u, i], dim=1)
        return self.fc(x)
        
data = [
    (0, 0, 5),
    (0, 2, 3),
    (1, 0, 4),
    (1, 2, 2),
    (2, 1, 5),
    (2, 2, 4),
]

users = torch.tensor([d[0] for d in data])
items = torch.tensor([d[1] for d in data])
ratings = torch.tensor([d[2] for d in data]).float().unsqueeze(1)

model = RecSysModel()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):

    preds = model(users, items)
    loss = criterion(preds, ratings)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(loss.item())
        
user_id = torch.tensor([0])
item_id = torch.tensor([1])  # unseen item

prediction = model(user_id, item_id)
print("Predicted rating:", prediction.item())