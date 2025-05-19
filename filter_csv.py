import pandas as pd
import os
import glob

# Paths to dataset files
BASE_PATH = 'dataset/CBIS-DDSM'  # Update with your dataset path
TRAIN_CSV = os.path.join(BASE_PATH, 'calc_case_description_train_set.csv')
TEST_CSV = os.path.join(BASE_PATH, 'calc_case_description_test_set.csv')
JPEG_PATH = os.path.join(BASE_PATH, 'jpeg')

# Get available SeriesInstanceUIDs from JPEG folder
def get_available_jpeg_series():
    available_series = set()
    for series_dir in glob.glob(os.path.join(JPEG_PATH, '1.3.6.1.4.1.9590*')):
        series_uid = os.path.basename(series_dir)
        if glob.glob(os.path.join(series_dir, '*.jpg')):
            available_series.add(series_uid)
    print(f"Found {len(available_series)} directories with JPEG files")
    return available_series

# Filter CSV to keep only rows with available JPEGs
def filter_csv(csv_file, available_series, output_file):
    try:
        df = pd.read_csv(csv_file)
        # Extract SeriesInstanceUID from image file path
        df['series_uid'] = df['image file path'].apply(lambda x: x.split('/')[-2])
        # Filter rows where series_uid is in available_series
        df_filtered = df[df['series_uid'].isin(available_series)]
        df_filtered = df_filtered.drop(columns=['series_uid'])  # Remove temporary column
        df_filtered.to_csv(output_file, index=False)
        print(f"Saved filtered CSV to {output_file} with {len(df_filtered)} rows (original: {len(df)})")
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

def main():
    # Get available JPEG series
    available_series = get_available_jpeg_series()
    if not available_series:
        print("No JPEG files found. Exiting.")
        return
    # Filter train and test CSVs
    filter_csv(TRAIN_CSV, available_series, os.path.join(BASE_PATH, 'calc_case_description_train_set_filtered.csv'))
    filter_csv(TEST_CSV, available_series, os.path.join(BASE_PATH, 'calc_case_description_test_set_filtered.csv'))

if __name__ == "__main__":
    main()