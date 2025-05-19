import pandas as pd
import numpy as np
import os
from PIL import Image
import warnings
import glob
warnings.filterwarnings('ignore')

# Paths to dataset files (update these to your local paths)
BASE_PATH = 'dataset/CBIS-DDSM'  # Update with your dataset path
TRAIN_CSV = os.path.join(BASE_PATH, 'calc_case_description_train_set_filtered.csv')  # Use filtered CSV if available
TEST_CSV = os.path.join(BASE_PATH, 'calc_case_description_test_set_filtered.csv')   # Use filtered CSV if available
DICOM_INFO_CSV = os.path.join(BASE_PATH, 'dicom_info.csv')
JPEG_PATH = os.path.join(BASE_PATH, 'jpeg')
OUTPUT_PATH = os.path.join(BASE_PATH, 'preprocessed')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Get available SeriesInstanceUIDs from JPEG folder
def get_available_jpeg_series():
    available_series = set()
    for series_dir in glob.glob(os.path.join(JPEG_PATH, '1.3.6.1.4.1.9590*')):
        series_uid = os.path.basename(series_dir)
        if glob.glob(os.path.join(series_dir, '*.jpg')):
            available_series.add(series_uid)
    print(f"Found {len(available_series)} directories with JPEG files")
    return available_series

# Load dicom_info.csv to map SeriesInstanceUID to JPEG paths
def load_dicom_info(available_series):
    try:
        dicom_df = pd.read_csv(DICOM_INFO_CSV)
        # Filter to include only SeriesInstanceUIDs with available JPEG files
        dicom_df = dicom_df[dicom_df['SeriesInstanceUID'].isin(available_series)]
        path_map = dict(zip(dicom_df['SeriesInstanceUID'], dicom_df['image_path']))
        print(f"Filtered dicom_info.csv to {len(path_map)} entries with available JPEG files")
        return path_map
    except Exception as e:
        print(f"Error loading dicom_info.csv: {e}")
        return {}

# Image preprocessing function
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Find JPEG file in a directory
def find_jpeg_file(series_dir):
    jpeg_files = glob.glob(os.path.join(series_dir, '*.jpg'))
    if jpeg_files:
        return jpeg_files[0]  # Return the first .jpg file found
    return None

# Load and preprocess dataset
def load_dataset(csv_file, image_base_path, path_map, available_series):
    # Check if filtered CSV exists, fall back to original if not
    if not os.path.exists(csv_file):
        print(f"Filtered CSV {csv_file} not found, trying original CSV")
        csv_file = csv_file.replace('_filtered', '')
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV {csv_file}: {e}")
        return np.array([]), np.array([]), np.array([])
    
    images = []
    labels = []
    metadata = []
    skipped = 0
    processed = 0
    processed_patients = set()
    
    for idx, row in df.iterrows():
        # Extract SeriesInstanceUID from image file path
        try:
            series_uid = row['image file path'].split('/')[-2]
        except IndexError:
            print(f"Invalid image file path format in row {idx}: {row['image file path']}")
            skipped += 1
            continue
        
        # Skip if SeriesInstanceUID is not in available JPEG files
        if series_uid not in available_series:
            print(f"SeriesInstanceUID {series_uid} not found in JPEG folder, skipping row {idx}")
            skipped += 1
            continue
        
        image_path = path_map.get(series_uid)
        if image_path is None:
            print(f"No image path found for SeriesInstanceUID {series_uid} in dicom_info.csv, row {idx}")
            skipped += 1
            continue
        
        # Construct series directory path
        series_dir = os.path.join(BASE_PATH, os.path.dirname(image_path.replace('CBIS-DDSM/jpeg/', 'jpeg/')))
        jpeg_path = find_jpeg_file(series_dir)
        
        if jpeg_path is None:
            print(f"No JPEG file found in directory {series_dir} for SeriesInstanceUID {series_uid}, row {idx}")
            skipped += 1
            continue
        
        img = load_and_preprocess_image(jpeg_path)
        if img is None:
            skipped += 1
            continue
        
        # Append image and label
        images.append(img)
        label = 0 if row['pathology'] in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK'] else 1
        labels.append(label)
        
        # Extract metadata (breast density, subtlety)
        meta = [row['breast density'], row['subtlety']]
        metadata.append(meta)
        processed += 1
        processed_patients.add(row['patient_id'])
        print(f"Processed image for SeriesInstanceUID {series_uid} in row {idx}, patient {row['patient_id']}")
    
    print(f"Processed {processed} images from {len(processed_patients)} patients, skipped {skipped} entries")
    return np.array(images), np.array(labels), np.array(metadata)

# Save preprocessed data
def save_preprocessed_data(images, labels, metadata, prefix):
    if len(images) == 0:
        print(f"No data to save for {prefix} dataset")
        return
    np.save(os.path.join(OUTPUT_PATH, f'{prefix}_images.npy'), images)
    np.save(os.path.join(OUTPUT_PATH, f'{prefix}_labels.npy'), labels)
    np.save(os.path.join(OUTPUT_PATH, f'{prefix}_metadata.npy'), metadata)
    print(f"Saved {prefix} data to {OUTPUT_PATH} with {len(images)} images")

# Main preprocessing function
def preprocess():
    # Get available SeriesInstanceUIDs from JPEG folder
    print("Scanning JPEG folder for available images...")
    available_series = get_available_jpeg_series()
    if not available_series:
        print("No JPEG files found. Exiting.")
        return
    
    # Load DICOM info for path mapping, filtered by available series
    print("Loading DICOM info...")
    path_map = load_dicom_info(available_series)
    if not path_map:
        print("Failed to load DICOM info. Exiting.")
        return
    
    # Process training data
    print("Preprocessing training data...")
    train_images, train_labels, train_metadata = load_dataset(TRAIN_CSV, JPEG_PATH, path_map, available_series)
    save_preprocessed_data(train_images, train_labels, train_metadata, 'train')
    
    # Process test data
    print("Preprocessing test data...")
    test_images, test_labels, test_metadata = load_dataset(TEST_CSV, JPEG_PATH, path_map, available_series)
    save_preprocessed_data(test_images, test_labels, test_metadata, 'test')

if __name__ == "__main__":
    preprocess()