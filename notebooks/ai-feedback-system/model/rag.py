documents = [
    "Our refund policy allows returns within 30 days.",
    "Shipping takes 3-5 business days.",
    "Customer support is available 24/7."
]

def retrieve(query):
    results = []

    for doc in documents:
        if any(word in doc.lower() for word in query.lower().split()):
            results.append(doc)

    return results
	
def rag_system(query):
    relevant_docs = retrieve(query)

    context = " ".join(relevant_docs)

    prompt = f"Context: {context}\n\nQuestion: {query}"

    return prompt
	
query = "What is your refund policy?"

context = rag_system(query)

print(context)