import torch
from transformers import BertTokenizer, BertForSequenceClassification

def get_up_down_trend_values(text):
    # Load the pre-trained model
    model_2 = BertForSequenceClassification.from_pretrained("Fine_tuned_model")
    model_2.to('cpu')

    # Tokenize the input text
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to('cpu')

    # Get model predictions
    outputs = model_2(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()

    # Extract upstream and downstream values
    uptrend_value, downtrend_value = predictions[0, 0], predictions[0, 1]

    return uptrend_value, downtrend_value

# Example usage
# text = "For the last quarter of 2010, Componenta's net sales doubled to EUR131m from EUR76m for the same period a year earlier, while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m"
# uptrend_value, downtrend_value = get_up_down_trend_values(text)

# print("X:", uptrend_value)
# print("Y:", downtrend_value)
