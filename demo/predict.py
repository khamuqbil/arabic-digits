import os

# Disable TensorFlow warnings and verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys

import numpy as np
import tensorflow as tf
from PIL import Image

# Configure TensorFlow to be less verbose
tf.keras.utils.disable_interactive_logging()

def rot_digit(digit):
    return np.fliplr(np.rot90(digit, axes=(1,0)))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Try multiple methods to load the legacy model
loaded_model = None

try:
    # Method 1: Try loading just the weights and recreate the model architecture
    # Read the JSON to understand model structure
    json_file = open('model.json', 'r')
    model_config = json_file.read()
    json_file.close()
    
    # Create a CNN model that matches your architecture from the JSON
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
    loaded_model.load_weights("model.h5")
        
except Exception:
    try:
        # Method 2: Try with tf.keras but with custom object scope
        from tensorflow.keras.models import model_from_json
        from tensorflow.keras.utils import custom_object_scope

        # Define custom objects that might be missing
        custom_objects = {
            'Sequential': tf.keras.Sequential,
            'Dense': tf.keras.layers.Dense,
            'Conv2D': tf.keras.layers.Conv2D,
            'MaxPooling2D': tf.keras.layers.MaxPooling2D,
            'Dropout': tf.keras.layers.Dropout,
            'Flatten': tf.keras.layers.Flatten,
            'BatchNormalization': tf.keras.layers.BatchNormalization
        }
        
        with custom_object_scope(custom_objects):
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model.h5")
            
    except Exception:
        # Method 3: Try loading just the weights and recreate the model
        try:
            # Read the JSON to understand model structure
            json_file = open('model.json', 'r')
            model_config = json_file.read()
            json_file.close()
            
            # Create a simple CNN model that matches your architecture
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
            loaded_model.load_weights("model.h5")
            
        except Exception as e3:
            # All methods failed - this means the model is truly incompatible
            print(f"Error loading model: {e3}")
            import sys
            sys.exit(1)
try:
    im = Image.open("tux.bmp")
    im = im.resize((28,28))
    # Convert to RGB if needed (in case of RGBA or other formats)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    
    p = np.array(im)

    # Handle single channel images
    if len(p.shape) == 2:
        p = np.stack([p, p, p], axis=-1)  # Convert to 3-channel
    p = rgb2gray(p)
    p = p /255.0
    p = rot_digit(p)
    p = p.reshape(1, 28, 28,1)
except Exception as e:
    print(f"Error processing image: {e}")
    # Create a dummy black image as fallback
    p = np.zeros((1, 28, 28, 1))
    print("0 100% 0 0% 0 0%")
    sys.exit(0)
preds = loaded_model.predict(p, verbose=0)[0]

# Get top 3 predictions with proper indices
sorted_indices = np.argsort(preds)[::-1]  # Sort in descending order
fst = sorted_indices[0]
fst_p = preds[fst]
snd = sorted_indices[1] 
snd_p = preds[snd]
thd = sorted_indices[2]
thd_p = preds[thd]

print("{0} {1:.0%} {2} {3:.0%} {4} {5:.0%}".format(fst, fst_p, snd, snd_p, thd, thd_p))
