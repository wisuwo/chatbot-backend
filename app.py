from flask import Flask, request, jsonify
from flask_cors import CORS
from chat import get_response

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return jsonify({"message": "Chatbot API running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("message", "")

    if not text:
        return jsonify({"error": "No message provided"}), 400

    response = get_response(text)
    return jsonify({"answer": response}), 200

if __name__ == "__main__":
    app.run(debug=True)
