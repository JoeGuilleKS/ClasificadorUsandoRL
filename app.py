from flask import Flask, request, jsonify
import joblib
import numpy as np
import cv2
import base64
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permite peticiones desde frontend local

model = joblib.load('modelo_logistico.pkl')

def preprocess_image(base64_str):
    img_str = re.search(r'base64,(.*)', base64_str).group(1)
    img_bytes = base64.b64decode(img_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten().astype(float) / 255
    return img.reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image']
    img_processed = preprocess_image(img_data)
    pred = model.predict(img_processed)[0]
    label = 'Perro' if pred == 1 else 'Gato'
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
