from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
model_path = r"D:\FYP\Streaming\my_sentiment_model"
# Try loading the tokenizer
try:
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForSequenceClassification.from_pretrained(model_path)
  # Model loaded successfully, proceed with your code here
except EnvironmentError as e:
  print("Error loading tokenizer:", e)
  # Handle t
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Now you can use the tokenizer
tokens = tokenizer.encode('This is a bad experience!', return_tensors='pt')
result = model(tokens)
sentiment = int(torch.argmax(result.logits))+1

# Print the predicted sentiment
print(f"Predicted sentiment: {sentiment}")
