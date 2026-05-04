import os
import torch
from model.model import LSTMClassifier

def load_trained_model(vocab_size):
    model = LSTMClassifier(vocab_size)

    path = "model/model.pth"

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Model file not found. Run: python model/train.py first."
        )

    model.load_state_dict(torch.load(path))
    model.eval()
    return model