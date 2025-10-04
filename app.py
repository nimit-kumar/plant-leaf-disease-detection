import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load model and class names
model = load_model('plant_leaf_disease_cnn_model.h5')
class_names = np.load('class_names.npy', allow_pickle=True).tolist()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Preprocess image
        img = image.load_img(filepath, target_size=(64, 64))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        # Prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100  # percent
        # Top-3 predictions
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_predictions = [
            {"class": class_names[i], "confidence": round(float(predictions[0][i]) * 100, 2)}
            for i in top3_indices
        ]
        result = {
            "predicted_class": class_names[predicted_class_idx],
            "confidence": round(confidence, 2),
            "top3": top3_predictions
        }
        return jsonify(result)
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7860, debug=True)
