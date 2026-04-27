import torch
import torch.nn as nn

sentence = ["not", "good"]

vocab = {"good":0, "movie":1, "not":2, "great":3, "film":4, "bad":5}

input_tensor = torch.tensor([[vocab[word] for word in sentence]])


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), 4)
        self.lstm = nn.LSTM(4, 8, batch_first=True)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        x = hidden[-1]
        x = torch.sigmoid(self.fc(x))
        return x
        

model = LSTMClassifier()

model.eval()

with torch.no_grad():
    prediction = model(input_tensor)

score = prediction.item()
print(score)


if score > 0.5:
    sentiment = "Positive"
else:
    sentiment = "Negative"


def respond(sentiment):
    if sentiment == "Positive":
        return "Glad you liked it!"
    else:
        return "Sorry to hear that. How can we improve?"

response = respond(sentiment)

print("Agent:", response)

def ai_agent(sentence):
    input_tensor = torch.tensor([[vocab[word] for word in sentence]])

    with torch.no_grad():
        score = model(input_tensor).item()

    sentiment = "Positive" if score > 0.5 else "Negative"

    return respond(sentiment)

print(ai_agent(["not", "good"]))
