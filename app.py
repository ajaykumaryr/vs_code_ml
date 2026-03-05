from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load('language_predictor_model.joblib')
le = joblib.load('label_encoder.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    pred = model.predict([text])
    proba = model.predict_proba([text]).max()
    language = le.inverse_transform(pred)[0]
    return jsonify({ "language": language, "confidence": float(proba) })

if __name__ == "__main__":
    app.run(debug=True, port=5000)