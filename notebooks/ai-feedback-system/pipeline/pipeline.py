import torch

def preprocess(text):
    return text.lower().split()


def predict_sentiment(model, vocab, words):
    indices = [vocab.get(word, 0) for word in words]
    input_tensor = torch.tensor([indices])

    with torch.no_grad():
        score = model(input_tensor).item()

    return "Positive" if score > 0.5 else "Negative"


def decision(sentiment):
    if sentiment == "Positive":
        return "No action needed"
    else:
        return "Flag for review"


def run_pipeline(model, vocab, text):
    words = preprocess(text)
    sentiment = predict_sentiment(model, vocab, words)
    action = decision(sentiment)

    return {
        "sentiment": sentiment,
        "action": action
    }