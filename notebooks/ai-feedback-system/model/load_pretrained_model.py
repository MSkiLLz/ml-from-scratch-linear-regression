from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("This product is amazing!")
print(result)
print(classifier("good movie"))
print(classifier("not good"))