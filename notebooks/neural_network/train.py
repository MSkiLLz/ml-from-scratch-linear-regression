import torch
import torch.nn as nn

movies = {
    "Movie A": [1, 0],  # action
    "Movie B": [0, 1],  # romance
    "Movie C": [1, 1],  # action + romance
}

user_profile = torch.tensor([1.0, 0.0])  # likes action

def similarity(user, item):
    return torch.dot(user, torch.tensor(item).float())
    
scores = {}

for movie, features in movies.items():
    scores[movie] = similarity(user_profile, features)

print(scores)

recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)

print("Recommendations:", recommended)

