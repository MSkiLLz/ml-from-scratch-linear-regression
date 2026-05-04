import torch
from model.model import LSTMClassifier

def load_trained_model(vocab_size):
    model = LSTMClassifier(vocab_size)
    model.load_state_dict(torch.load("model/model.pth"))
    model.eval()
    return model