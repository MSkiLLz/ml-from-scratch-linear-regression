memory = []

def store_interaction(text, sentiment):
    memory.append({
        "text": text,
        "sentiment": sentiment
    })


def analyze_memory():
    positives = sum(1 for m in memory if m["sentiment"] == "Positive")
    negatives = sum(1 for m in memory if m["sentiment"] == "Negative")
    return positives, negatives


def agent_response(sentiment):
    positives, negatives = analyze_memory()

    if negatives > positives:
        return "We’ve noticed repeated issues. Escalating this."

    if sentiment == "Positive":
        return "Thanks for your feedback!"
    else:
        return "Sorry about that. We'll work on improving."