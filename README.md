# Arabic Digits Recognition

A machine learning model that classifies handwritten Arabic digits (٠-٩). Built with convolutional neural networks and deployed as a web application for demonstration purposes.

## Model Architecture

- **Network Type**: Convolutional Neural Network (CNN)
- **Layers**: 5 convolutional layers with max pooling and dropout
- **Regularization**: Batch normalization and dropout (0.2-0.25)
- **Activation**: ReLU activation functions, softmax output
- **Input**: 28×28 grayscale images, normalized to [0,1]
- **Output**: 10-class softmax probability distribution
- **Training**: 20 epochs with learning rate reduction on plateau

## Demo Application

**Live Demo**: [http://arabic-digits-app.qatarcentral.azurecontainer.io](http://arabic-digits-app.qatarcentral.azurecontainer.io)

- **Web Interface**: Draw digits on canvas for classification
- **API**: `/api/predict` endpoint for programmatic use
- **Results**: Shows top 3 predictions with confidence scores
- **Audio**: Arabic pronunciation of recognized digits

## Technology Stack

- **Backend**: Flask (Python)
- **ML Framework**: TensorFlow/Keras
- **Frontend**: HTML5 Canvas, JavaScript, Spectre.css
- **Deployment**: Docker, Azure Container Instances
- **CI/CD**: GitHub Actions

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Using Nix

```bash
nix-shell --run "pip install -r requirements.txt && python app.py"
```

### Docker

```bash
# Build the image
docker build -t arabic-digits .

# Run the container
docker run -p 80:80 arabic-digits
```

## API Usage

### Predict Endpoint

**POST** `/api/predict`

```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Response:**

```json
{
  "predictions": [
    { "digit": 5, "confidence": 0.95 },
    { "digit": 3, "confidence": 0.03 },
    { "digit": 8, "confidence": 0.02 }
  ]
}
```

## Project Structure

```
.
├── app.py              # Main Flask application
├── demo/               # Demo assets and legacy code
│   ├── main.py        # Alternative Flask app
│   ├── model.h5       # Trained CNN model weights
│   ├── model.json     # Model architecture
│   ├── predict.py     # Prediction logic
│   ├── static/        # CSS, JS, audio files
│   └── templates/     # HTML templates
├── Dockerfile         # Container configuration
├── requirements.txt   # Python dependencies
└── .github/workflows/ # CI/CD pipeline
```

## Model Details

- **Architecture**: Multi-layer CNN with dropout and batch normalization
- **Input**: 28x28 grayscale images
- **Output**: 10 classes (digits 0-9)
- **Training Data**: Arabic handwritten digits dataset
- **Accuracy**: High confidence recognition with uncertainty quantification

## Dataset

The model was trained on **MADBase (Modified Arabic Digits Database)** featuring:

- **Dataset**: MADBase - 70,000 total images
- **Training Set**: 60,000 images
- **Testing Set**: 10,000 images
- **Format**: 28x28 grayscale images (784 pixels total)
- **Classes**: 10 Arabic-Indic digit classes (٠-٩)
- **Preprocessing**: Normalized pixel values (0-255 → 0-1), rotation correction
- **Source**: Electronics Engineering Department, American University in Cairo

### Model Performance

- **Test Accuracy**: 99.35% (without data augmentation)
- **Enhanced Accuracy**: 99.75% (with data augmentation)
- **Architecture**: Multi-layer CNN with 5 convolutional layers, dropout, and batch normalization
- **Training**: 20 epochs with learning rate reduction on plateau

### Dataset Reference

**MADBase**: Sherif Abdelazeem, Ezzat Al-Sherif. "Arabic handwritten digit recognition." _International Journal of Document Analysis and Recognition (IJDAR)_, 11(3):127–141, 2008.

**Original Research**: Khalid Alkhaldi. "Handwritten Arabic Digits Recognition Using Neural Networks." Supervised by Dr. Nasser Alshammari, College of Computer and Information Sciences, Jouf University, April 2019.

If you use this model in your research, please cite:

```bibtex
@article{abdelazeem2008arabic,
  title={Arabic handwritten digit recognition},
  author={Abdelazeem, Sherif and El-Sherif, Ezzat},
  journal={International Journal of Document Analysis and Recognition (IJDAR)},
  volume={11},
  number={3},
  pages={127--141},
  year={2008}
}
```

## Acknowledgments

Special thanks to **Dr. Nasser Alshammari** for his invaluable guidance, supervision, and support throughout this research project at Jouf University. His expertise and mentorship were instrumental in achieving these state-of-the-art results in Arabic handwritten digit recognition.
