from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HEADERS = {"Authorization": "Bearer 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"}

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    text = data["text"]
    response = requests.post(HUGGING_FACE_API_URL, headers=HEADERS, json={"inputs": text})
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)