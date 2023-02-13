import json
import numpy as np
import tensorflow as tf
from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def masuk():
    return "MASUK"

MODEL_PATH = "../serving_model_dir/diabetes-detection-model/1676195266"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=['POST'])
def predict():
    req_json = request.json
    
    data = req_json.get("data")

    inputs = [np.array([x]) for x in data[0]]

    prediction = model.predict(inputs)

    if prediction[0][0] > .8:
        prediction = "diagnosed diabet"
    else:
        prediction = "not diagnosed diabet"

    response = {
        "prediction": prediction
    }
 
    return json.dumps(response)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)