import json
import numpy as np
import tensorflow as tf
from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def masuk():
    return "MASUK"

MODEL_PATH = "../serving_model_dir/wine-detection-model/1675868727"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=['POST'])
def predict():
    req_json = request.json
    
    data = req_json.get("data")

    prediction = model.predict(data)

    response = {
        "prediction": prediction
    }

    return json.dumps(response)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)