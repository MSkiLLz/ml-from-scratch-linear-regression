import torch
import torch.nn as nn

def preprocess(text):
    return text.lower()
    
def predict_sentiment(sentence):
    input_tensor = torch.tensor([[vocab[word] for word in sentence]])

    with torch.no_grad():
        score = model(input_tensor).item()

    return "Positive" if score > 0.5 else "Negative"

def ai_pipeline(text):
    processed = preprocess(text)
    words = processed.split()

    sentiment = predict_sentiment(words)
    action = decision(sentiment)

    return {
        "sentiment": sentiment,
        "action": action
    }


result = ai_pipeline("This product is not good")
print(result)