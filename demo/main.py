import base64
import logging
import os
import re
import subprocess
from io import BytesIO

from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/health')
def health_check():
    """Health check endpoint for Azure"""
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route('/save', methods=['POST'])
def save():
    try:
        digit = re.sub('^data:image/.+;base64,', '', request.form['digit'])
        # Add padding if needed for base64 decoding
        missing_padding = len(digit) % 4
        if missing_padding:
            digit += '=' * (4 - missing_padding)
            
        with Image.open(BytesIO(base64.b64decode(digit))) as im:
            im.save("tux.bmp")
            im.close()
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        predict_script = os.path.join(script_dir, 'predict.py')
        result = subprocess.run(['python', predict_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=script_dir)
        if result.returncode == 0:
            prediction = result.stdout.decode('utf-8')
            logger.info(f"Prediction successful: {prediction}")
            return prediction
        else:
            logger.error(f"Prediction failed: {result.stderr.decode('utf-8')}")
            return jsonify({"error": "Prediction failed"}), 500
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
        
        digit = re.sub('^data:image/.+;base64,', '', data['image'])
        # Add padding if needed for base64 decoding
        missing_padding = len(digit) % 4
        if missing_padding:
            digit += '=' * (4 - missing_padding)
            
        with Image.open(BytesIO(base64.b64decode(digit))) as im:
            im.save("tux.bmp")
            im.close()
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        predict_script = os.path.join(script_dir, 'predict.py')
        result = subprocess.run(['python', predict_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=script_dir)
        if result.returncode == 0:
            prediction = result.stdout.decode('utf-8').strip()
            # Parse the prediction format: "5 95% 6 3% 8 1%"
            parts = prediction.split()
            predictions = []
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    digit_num = int(parts[i])
                    confidence = float(parts[i + 1].replace('%', '')) / 100
                    predictions.append({"digit": digit_num, "confidence": confidence})
            
            return jsonify({"predictions": predictions})
        else:
            logger.error(f"Prediction failed: {result.stderr.decode('utf-8')}")
            return jsonify({"error": "Prediction failed"}), 500
            
    except Exception as e:
        logger.error(f"Error in API prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port, debug=False)
