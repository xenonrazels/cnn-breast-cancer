import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Paths to preprocessed data and model outputs
BASE_PATH = 'dataset/CBIS-DDSM'  # Path to CBIS-DDSM dataset
PREPROCESSED_PATH = os.path.join(BASE_PATH, 'preprocessed')
MODEL_PATH = 'models'  # Models saved outside dataset/, at project level
os.makedirs(MODEL_PATH, exist_ok=True)

# Load preprocessed data
def load_preprocessed_data(prefix):
    try:
        images = np.load(os.path.join(PREPROCESSED_PATH, f'{prefix}_images.npy'))
        labels = np.load(os.path.join(PREPROCESSED_PATH, f'{prefix}_labels.npy'))
        metadata = np.load(os.path.join(PREPROCESSED_PATH, f'{prefix}_metadata.npy'))
        print(f"Loaded {prefix} data: {len(images)} images")
        return images, labels, metadata
    except Exception as e:
        print(f"Error loading {prefix} data: {e}")
        return np.array([]), np.array([]), np.array([])

# Build CNN model for feature extraction
def build_cnn_model(input_shape=(224, 224, 1)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5)
    ])
    return model

# Extract features using CNN
def extract_cnn_features(model, images):
    if len(images) == 0:
        print("No images to extract features from")
        return np.array([])
    features = model.predict(images, batch_size=32, verbose=1)
    return features

# Train and evaluate models
def train_and_evaluate():
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_images, train_labels, train_metadata = load_preprocessed_data('train')
    test_images, test_labels, test_metadata = load_preprocessed_data('test')
    
    if len(train_images) == 0 or len(test_images) == 0:
        print("Empty training or test dataset. Exiting.")
        return
    
    # Add channel dimension for CNN
    train_images = train_images[..., np.newaxis]
    test_images = test_images[..., np.newaxis]
    
    # Build CNN model
    print("Building CNN model...")
    cnn_model = build_cnn_model()
    
    # Extract features
    print("Extracting CNN features...")
    train_features = extract_cnn_features(cnn_model, train_images)
    test_features = extract_cnn_features(cnn_model, test_images)
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Failed to extract features. Exiting.")
        return
    
    # Combine CNN features with metadata
    train_features_combined = np.hstack((train_features, train_metadata))
    test_features_combined = np.hstack((test_features, test_metadata))
    
    # Standardize features
    scaler = StandardScaler()
    train_features_combined = scaler.fit_transform(train_features_combined)
    test_features_combined = scaler.transform(test_features_combined)
    
    # Save scaler
    scaler_path = os.path.join(MODEL_PATH, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Saved StandardScaler to {scaler_path}")
    
    # Linear Regression
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(train_features_combined, train_labels)
    lr_predictions = lr_model.predict(test_features_combined)
    lr_predictions_binary = (lr_predictions > 0.5).astype(int)
    
    # Save Linear Regression model
    lr_model_path = os.path.join(MODEL_PATH, 'lr_model.joblib')
    joblib.dump(lr_model, lr_model_path)
    print(f"Saved Linear Regression model to {lr_model_path}")
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(train_features_combined, train_labels)
    rf_predictions = rf_model.predict(test_features_combined)
    
    # Save Random Forest model
    rf_model_path = os.path.join(MODEL_PATH, 'rf_model.joblib')
    joblib.dump(rf_model, rf_model_path)
    print(f"Saved Random Forest model to {rf_model_path}")
    
    # Save CNN model as model.h5
    cnn_model_path = os.path.join(MODEL_PATH, 'model.h5')
    cnn_model.save(cnn_model_path)
    print(f"Saved CNN model to {cnn_model_path}")
    
    # Evaluate models
    print("\nEvaluation Results:")
    print("Linear Regression:")
    print(f"Accuracy: {accuracy_score(test_labels, lr_predictions_binary):.4f}")
    print(f"Precision: {precision_score(test_labels, lr_predictions_binary, zero_division=0):.4f}")
    print(f"Recall: {recall_score(test_labels, lr_predictions_binary, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(test_labels, lr_predictions_binary, zero_division=0):.4f}")
    
    print("\nRandom Forest:")
    print(f"Accuracy: {accuracy_score(test_labels, rf_predictions):.4f}")
    print(f"Precision: {precision_score(test_labels, rf_predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(test_labels, rf_predictions, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(test_labels, rf_predictions, zero_division=0):.4f}")

if __name__ == "__main__":
    train_and_evaluate()