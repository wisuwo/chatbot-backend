import torch
import json
import random  # Ensure random is imported
from flask import Flask, request, jsonify
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Enable CORS for all routes

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

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
                response = random.choice(intent["responses"])
                return jsonify({"answer": response})

        return jsonify({"answer": "I'm not sure how to respond to that."})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)
