import torch

ratings = torch.tensor([
    [5, 0, 3],  # User 1
    [4, 0, 2],  # User 2
    [0, 5, 4],  # User 3
], dtype=torch.float)

user_profile = torch.tensor([1.0, 0.0])  # likes action

def similarity(u1, u2):
    return torch.dot(u1, u2)

sim_1_2 = similarity(ratings[0], ratings[1])
sim_1_3 = similarity(ratings[0], ratings[2])

print(sim_1_2, sim_1_3)

target_user = ratings[0]

scores = torch.zeros(3)

for i in range(len(ratings)):
    if i != 0:
        sim = similarity(target_user, ratings[i])
        scores += sim * ratings[i]

print(scores)



