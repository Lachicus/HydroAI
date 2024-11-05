# app.py
from flask import Flask, request, render_template, jsonify, url_for
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Define paths to models and encoders
MODELS = {
    "Davinci-004-v6i": "models/ediboiAI-turing-003-v5i.keras",
    "Turing-003-v5i": "models/ediboiAI-turing-003-v5i.keras",
    "Turing-003-v6": "models/ediboiAI-turing-003-v5i.keras",
    "Turing-003-v6i": "models/ediboiAI-turing-003-v5i.keras"
}
LABEL_ENCODER_TYPE_PATH = "label_encoders/turing_type_encoder.pkl"
LABEL_ENCODER_WATER_LEVEL_PATH = "label_encoders/turing_water_level_encoder.pkl"

# Load label encoders
le_type = joblib.load(LABEL_ENCODER_TYPE_PATH)
le_water_level = joblib.load(LABEL_ENCODER_WATER_LEVEL_PATH)

img_height, img_width = 224, 224

# Function to load the selected model
def load_selected_model(model_key):
    model_path = MODELS.get(model_key)
    return load_model(model_path) if model_path else None

# Prediction function
def get_allowed_vehicles(class_label):
    passability = {
        'Green': ["All Vehicles"],
        'Yellow': ["Sedan", "Truck", "Motorcycle", "Bus"],
        'Orange': ["Truck", "Bus"],
        'Red': ["No Vehicle is Advicable to Pass"]
    }
    return passability.get(class_label, [])

def predict_image(image, model):
    img = img_to_array(image.resize((img_width, img_height))) / 255.0
    img_array = np.expand_dims(img, axis=0)

    preds = model.predict(img_array)
    type_pred = preds[0]
    water_level_pred = preds[1]

    type_idx = np.argmax(type_pred, axis=1)[0]
    water_level_idx = np.argmax(water_level_pred, axis=1)[0]

    class_label_type = le_type.inverse_transform([type_idx])[0]
    class_label_water_level = le_water_level.inverse_transform([water_level_idx])[0]
    allowed_vehicles = get_allowed_vehicles(class_label_type)

    return class_label_type, class_label_water_level, allowed_vehicles

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'No file or model selected'}), 400

    model_key = request.form['model']
    model = load_selected_model(model_key)
    if not model:
        return jsonify({'error': 'Invalid model selected'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    class_label_type, class_label_water_level, allowed_vehicles = predict_image(image, model)

    result = {
        "flood_label": class_label_type,
        "water_level": class_label_water_level,
        "allowed_vehicles": allowed_vehicles
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
