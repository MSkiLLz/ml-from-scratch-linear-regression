import torch

num_users = 3
num_items = 3
embedding_dim = 2

user_emb = torch.randn(num_users, embedding_dim, requires_grad=True)
item_emb = torch.randn(num_items, embedding_dim, requires_grad=True)

def predict(u, i):
    return torch.dot(user_emb[u], item_emb[i])


ratings = torch.tensor([
    [5, 0, 3],
    [4, 0, 2],
    [0, 5, 4]
], dtype=torch.float)

loss = 0

for u in range(num_users):
    for i in range(num_items):
        if ratings[u, i] > 0:
            pred = predict(u, i)
            loss += (pred - ratings[u, i])**2
            

optimizer = torch.optim.SGD([user_emb, item_emb], lr=0.01)

for epoch in range(100):

    loss = 0

    for u in range(num_users):
        for i in range(num_items):
            if ratings[u, i] > 0:
                pred = predict(u, i)
                loss += (pred - ratings[u, i])**2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(loss.item())
        
user_id = 0

scores = [predict(user_id, i).item() for i in range(num_items)]

print(scores)