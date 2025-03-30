import torch
import json
import random
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)

# Load the trained model
FILE = "training_data/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()  # Set model to evaluation mode

# ✅ This is the function app.py will import
def get_response(user_message):
    # Process input
    sentence = tokenize(user_message)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    # Predict intent
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Find response
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I'm not sure how to respond to that."
