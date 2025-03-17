import torch
import json
import random
from model.model import NeuralNet
from model.nltk_utils import tokenize, bag_of_words

# Cache model to avoid reloading
model = None
all_words, tags, intents = None, None, None

def load_model():
    global model, all_words, tags, intents
    if model is None:
        with open("model/intents.json", "r") as f:
            intents = json.load(f)
        FILE = "training_data/data.pth"
        data = torch.load(FILE)
        model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
        model.load_state_dict(data["model_state"])
        model.eval()
        all_words = data["all_words"]
        tags = data["tags"]

def predict_intent(message):
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "I'm not sure how to respond to that."

def handler(request):
    load_model()
    if request.method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            "body": "",
        }

    if request.method == "POST":
        try:
            data = request.get_json()
            message = data.get("message", "")
            if not message:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": "No message provided"}),
                    "headers": {"Access-Control-Allow-Origin": "*"},
                }
            response = predict_intent(message)
            return {
                "statusCode": 200,
                "body": json.dumps({"answer": response}),
                "headers": {"Access-Control-Allow-Origin": "*"},
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": str(e)}),
                "headers": {"Access-Control-Allow-Origin": "*"},
            }

    return {
        "statusCode": 405,
        "body": json.dumps({"error": "Method Not Allowed"}),
        "headers": {"Access-Control-Allow-Origin": "*"},
    }
