import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 4)
        self.lstm = nn.LSTM(4, 8, batch_first=True)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = torch.sigmoid(self.fc(x))
        return x