import os
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder="build", static_url_path="/")
CORS(app)

@app.route("/")
def serve_react():
    return send_from_directory(app.static_folder, "index.html")

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
