import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Model load karo
MODEL_PATH = r"C:\Users\nimit\Music\.vscode\plant_leaf_disease_cnn_model.h5"
model = load_model(MODEL_PATH)

# Apne class labels (aap training ke time ki classes likho)
class_names = ["Healthy", "Powdery", "Rust"]  # <-- apni classes dalna

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    if file:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(128, 128))  # apne training ke size pe dhyan do
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return jsonify({
            "class": predicted_class,
            "confidence": round(confidence * 100, 2)
        })

if __name__ == '__main__':
    app.run(debug=True)
