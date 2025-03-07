from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv() 

app = Flask(__name__)
CORS(app)

HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

@app.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.json
    text = data["text"]
    response = requests.post(HUGGING_FACE_API_URL, headers=HEADERS, json={"inputs": text})
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)
