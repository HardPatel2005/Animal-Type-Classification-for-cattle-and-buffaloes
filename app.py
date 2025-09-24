import numpy as np
import tensorflow as tf
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import base64
import gdown

app = Flask(__name__)

# --- Model & Configuration ---
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Google Drive model info
SAVED_MODEL_PATH = 'models/fine_tuned_model_mobilenet_v2_fine_tuned.keras'
MODEL_GDRIVE_ID = '1XEiUrq7iT9ukPr4WF4q7iyORRa9s5-Vx'
MODEL_DOWNLOAD_URL = f'https://drive.google.com/uc?id={MODEL_GDRIVE_ID}'

# Download model if not already present
os.makedirs('models', exist_ok=True)
if not os.path.exists(SAVED_MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_DOWNLOAD_URL, SAVED_MODEL_PATH, quiet=False)

# Load model
try:
    trained_model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    trained_model = None

# Labels
LABELS_CSV_PATH = 'labels.csv'
class_labels = None
if os.path.exists(LABELS_CSV_PATH):
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    class_labels = sorted(labels_df['label'].unique().tolist())
    print("Class labels loaded.")
else:
    print(f"Labels CSV not found at {LABELS_CSV_PATH}. Make sure it's in the same directory.")

# Buffalo breeds for type classification
buffalo_breeds = [
    'Banni', 'Bhadawari', 'Murrah',
    'Jaffrabadi buffalo', 'Jaffrabadi', 'Murrah buffalo', 'Mehsana',
    'Nagpuri', 'Nili_Ravi', 'Surti', 'Toda'
]

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_for_model(image_path, img_size=(224, 224)):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=img_size)
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict_breed(model, image_path, class_labels):
    if model is None or class_labels is None:
        return "Error: Model or labels not loaded.", 0.0
    img = process_image_for_model(image_path)
    if img is None:
        return "Error: Image processing failed.", 0.0
    img = tf.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    return predicted_label, confidence

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if trained_model is None or class_labels is None:
        error_message = "Application error: Model or class labels are not loaded. Check the console for details."
        return render_template('result.html', data={'error': True, 'message': error_message})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_breed, confidence = predict_breed(trained_model, filepath, class_labels)
        animal_type = "Buffalo" if predicted_breed in buffalo_breeds else "Cow"
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        data = {
            'error': False,
            'image_data': f'data:image/jpeg;base64,{encoded_string}',
            'filename': filename,
            'breed': predicted_breed,
            'type': animal_type,
            'confidence': f"{confidence:.2f}%"
        }
        return render_template('result.html', data=data)
    else:
        error_message = "Invalid file type. Please upload a .png, .jpg, or .jpeg file."
        return render_template('result.html', data={'error': True, 'message': error_message})

if __name__ == '__main__':
    app.run(debug=True)
