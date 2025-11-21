import base64
import os
import re
from io import BytesIO

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

# Disable TensorFlow warnings and verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging

import tensorflow as tf

# Configure TensorFlow to be less verbose
tf.keras.utils.disable_interactive_logging()

app = Flask(__name__, template_folder='demo/templates', static_folder='demo/static')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the loaded model
loaded_model = None

def load_model():
    """Load the trained model"""
    global loaded_model
    try:
        # Method 1: Try loading just the weights and recreate the model architecture
        # Read the JSON to understand model structure
        with open('demo/model.json', 'r') as json_file:
            model_config = json_file.read()
        
        # Create a CNN model that matches your architecture
        loaded_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Try to load the weights
        loaded_model.load_weights("demo/model.h5")
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def rot_digit(digit):
    """Rotate digit image"""
    return np.fliplr(np.rot90(digit, axes=(1,0)))

def rgb2gray(rgb):
    """Convert RGB to grayscale"""
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def preprocess_image(image_data):
    """Preprocess the image for prediction"""
    try:
        # Decode base64 image
        digit = re.sub('^data:image/.+;base64,', '', image_data)
        image = Image.open(BytesIO(base64.b64decode(digit)))
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_array = rgb2gray(img_array)
        
        # Normalize
        img_array = img_array / 255.0
        
        # Rotate digit
        img_array = rot_digit(img_array)
        
        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template("index.html")

@app.route('/health')
def health_check():
    """Health check endpoint for Azure"""
    return jsonify({"status": "healthy", "model_loaded": loaded_model is not None})

@app.route('/save', methods=['POST'])
def predict():
    """Predict Arabic digit from drawn image"""
    try:
        if 'digit' not in request.form:
            return jsonify({"error": "No image data provided"}), 400
        
        if loaded_model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Preprocess image
        img_array = preprocess_image(request.form['digit'])
        if img_array is None:
            return jsonify({"error": "Error processing image"}), 400
        
        # Make prediction
        predictions = loaded_model.predict(img_array, verbose=0)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "digit": int(idx),
                "confidence": float(predictions[idx])
            })
        
        # Format response (keeping compatibility with existing frontend)
        response = f"{results[0]['digit']} {results[0]['confidence']:.0%} {results[1]['digit']} {results[1]['confidence']:.0%} {results[2]['digit']} {results[2]['confidence']:.0%}"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for prediction"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        if loaded_model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Preprocess image
        img_array = preprocess_image(data['image'])
        if img_array is None:
            return jsonify({"error": "Error processing image"}), 400
        
        # Make prediction
        predictions = loaded_model.predict(img_array, verbose=0)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "digit": int(idx),
                "confidence": float(predictions[idx])
            })
        
        return jsonify({"predictions": results})
        
    except Exception as e:
        logger.error(f"Error in API prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        logger.error("Failed to load model. Exiting...")
        exit(1)
    
    # Run the app
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # Load model when running with WSGI server (like gunicorn)
    load_model()
