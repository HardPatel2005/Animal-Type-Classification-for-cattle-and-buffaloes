import numpy as np
import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)

# --- Model & Configuration ---
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to your saved model and labels file
SAVED_MODEL_PATH = 'models/fine_tuned_model_mobilenet_v2_fine_tuned.keras'
LABELS_CSV_PATH = 'labels.csv'

# Check if model exists and load it
try:
    if not os.path.exists(SAVED_MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {SAVED_MODEL_PATH}")
    trained_model = load_model(SAVED_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    trained_model = None

# Get class labels from labels.csv
class_labels = None
if os.path.exists(LABELS_CSV_PATH):
    labels_df = pd.read_csv(LABELS_CSV_PATH)
    class_labels = sorted(labels_df['label'].unique().tolist())
    print("Class labels loaded.")
else:
    print(f"Labels CSV not found at {LABELS_CSV_PATH}. Make sure it's in the same directory.")

# List of buffalo breeds for classification
buffalo_breeds = [
    'Banni', 'Bhadawari', 'Murrah',
    'Jaffrabadi buffalo', 'Jaffrabadi', 'Murrah buffalo', 'Mehsana',
    'Nagpuri', 'Nili_Ravi', 'Surti', 'Toda'
]

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image_for_model(image_path, img_size=(224, 224)):
    try:
        # Read the image file
        image = tf.io.read_file(image_path)
        # Decode the image with 3 channels
        image = tf.image.decode_image(image, channels=3)
        # Convert to float and resize
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=img_size)
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict_breed(model, image_path, class_labels):
    if model is None or class_labels is None:
        return "Error: Model or labels not loaded."
    
    img = process_image_for_model(image_path)
    if img is None:
        return "Error: Image processing failed."
        
    img = tf.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class_index]
    
    # Get the prediction confidence/score
    confidence = predictions[0][predicted_class_index] * 100
    
    return predicted_label, confidence

# --- Flask Routes ---
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
        
        # Save the uploaded file
        file.save(filepath)

        # Make prediction
        predicted_breed, confidence = predict_breed(trained_model, filepath, class_labels)
        
        # Determine animal type
        animal_type = "Buffalo" if predicted_breed in buffalo_breeds else "Cow"
        
        # Encode image to display on the page
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare data for the template
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