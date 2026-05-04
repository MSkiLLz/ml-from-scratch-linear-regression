from model.load_model import load_trained_model
from pipeline.pipeline import run_pipeline
from agent.agent import store_interaction, agent_response

# Simple vocab (you can improve later)
vocab = {
    "good": 0,
    "bad": 1,
    "great": 2,
    "terrible": 3,
    "not": 4
}

model = load_trained_model(len(vocab))


def main():
    text = input("Enter customer feedback: ")

    result = run_pipeline(model, vocab, text)

    store_interaction(text, result["sentiment"])

    response = agent_response(result["sentiment"])

    print("\n--- Result ---")
    print(result)
    print("Agent:", response)


if __name__ == "__main__":
    main()