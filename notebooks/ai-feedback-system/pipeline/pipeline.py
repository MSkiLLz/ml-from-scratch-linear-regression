import torch

def preprocess(text):
    return text.lower().split()


def predict_sentiment(text):
    result = classifier(text)[0]
    return "Positive" if result["label"] == "POSITIVE" else "Negative"


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