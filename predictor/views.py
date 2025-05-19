from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
import os
from pathlib import Path

# Paths to saved models
MODEL_PATH = Path('models')  # Models saved outside dataset/, at project level
CNN_MODEL_PATH = MODEL_PATH / 'model.h5'
RF_MODEL_PATH = MODEL_PATH / 'rf_model.joblib'
SCALER_PATH = MODEL_PATH / 'scaler.joblib'

# Load models
try:
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    cnn_model = None
    rf_model = None
    scaler = None

# Preprocess image
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        img_array = img_array[..., np.newaxis]  # Add channel dimension
        return img_array, None
    except Exception as e:
        return None, str(e)

# View for handling image upload and prediction
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        if not image_file.name.lower().endswith('.jpg'):
            return render(request, 'predictor/upload.html', {'error': 'Please upload a JPEG image'})
        
        # Save uploaded image temporarily
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        uploaded_file_path = fs.path(filename)
        
        # Preprocess image
        img_array, error = preprocess_image(uploaded_file_path)
        if img_array is None:
            fs.delete(filename)
            return render(request, 'predictor/upload.html', {'error': f'Image processing error: {error}'})
        
        # Check if models are loaded
        if cnn_model is None or rf_model is None or scaler is None:
            fs.delete(filename)
            return render(request, 'predictor/upload.html', {'error': 'Model loading failed. Please ensure models are trained and saved.'})
        
        # Extract CNN features
        try:
            features = cnn_model.predict(np.array([img_array]), batch_size=1, verbose=0)
        except Exception as e:
            fs.delete(filename)
            return render(request, 'predictor/upload.html', {'error': f'Feature extraction error: {e}'})
        
        # Use dummy metadata (mean values from training)
        dummy_metadata = np.array([[3.0, 3.0]])  # Example: mean breast density=3, subtlety=3
        features_combined = np.hstack((features, dummy_metadata))
        
        # Standardize features
        try:
            features_combined = scaler.transform(features_combined)
        except Exception as e:
            fs.delete(filename)
            return render(request, 'predictor/upload.html', {'error': f'Scaling error: {e}'})
        
        # Predict using Random Forest
        try:
            prediction = rf_model.predict(features_combined)[0]
            result = 'Malignant' if prediction == 1 else 'Benign'
        except Exception as e:
            fs.delete(filename)
            return render(request, 'predictor/upload.html', {'error': f'Prediction error: {e}'})
        
        # Clean up
        fs.delete(filename)
        return render(request, 'predictor/upload.html', {'result': result})
    
    return render(request, 'predictor/upload.html')