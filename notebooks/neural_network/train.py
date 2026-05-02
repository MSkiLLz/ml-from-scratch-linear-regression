import torch
import torch.nn as nn

memory = []

def store_interaction(user_input, sentiment):
    memory.append({
        "input": user_input,
        "sentiment": sentiment
    })
    
def analyze_memory():
    negatives = sum(1 for m in memory if m["sentiment"] == "Negative")
    positives = sum(1 for m in memory if m["sentiment"] == "Positive")

    return positives, negatives
    
def respond(sentiment):
    positives, negatives = analyze_memory()

    if negatives > positives:
        return "I see you've had several bad experiences. Let's fix that."

    if sentiment == "Positive":
        return "Glad you liked it!"
    else:
        return "Sorry to hear that. How can we improve?"

def ai_agent(sentence):
    input_tensor = torch.tensor([[vocab[word] for word in sentence]])

    with torch.no_grad():
        score = model(input_tensor).item()

    sentiment = "Positive" if score > 0.5 else "Negative"

    store_interaction(sentence, sentiment)

    return respond(sentiment)