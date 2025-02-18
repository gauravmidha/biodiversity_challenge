import os
import pandas as pd
import numpy as np
import rasterio
from sklearn.impute import SimpleImputer

# Define paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/')) + "/"
RASTER_FILE = DATA_PATH + "TerraClimate_output.tiff"

def load_data():
    """Load training and validation datasets."""
    train = pd.read_csv(DATA_PATH + "Training_Data.csv")
    valid = pd.read_csv(DATA_PATH + "Validation_Template.csv")
    return train, valid

def extract_climate_features(df):
    """Extract climate features from TerraClimate GeoTIFF."""
    print("🌍 Extracting climate features...")

    with rasterio.open(RASTER_FILE) as dataset:
        transform = dataset.transform
        raster_data = dataset.read()

        extracted_features = []
        for _, row in df.iterrows():
            lon, lat = row["Longitude"], row["Latitude"]
            row_idx, col_idx = ~transform * (lon, lat)
            row_idx, col_idx = int(row_idx), int(col_idx)

            if 0 <= row_idx < dataset.height and 0 <= col_idx < dataset.width:
                pixel_values = raster_data[:, row_idx, col_idx]
            else:
                pixel_values = [np.nan] * dataset.count  # Assign NaN for out-of-bounds

            extracted_features.append(pixel_values)

    feature_columns = [f"climate_var_{i+1}" for i in range(raster_data.shape[0])]
    return pd.DataFrame(extracted_features, columns=feature_columns)

def handle_missing_values(df):
    """Impute missing values in climate features using median strategy."""
    print("🔍 Checking and handling missing values...")

    # Select only climate feature columns
    feature_columns = [col for col in df.columns if "climate_var" in col]

    # Print missing value count before imputation
    print("🔹 Missing values before imputation:\n", df[feature_columns].isnull().sum())

    # Impute missing values with the median
    imputer = SimpleImputer(strategy="median")
    df[feature_columns] = imputer.fit_transform(df[feature_columns])

    # Print missing value count after imputation
    print("✅ Missing values handled successfully!")

    return df

if __name__ == "__main__":
    # Load Data
    train, valid = load_data()

    # Extract Climate Features
    train_features = extract_climate_features(train)
    valid_features = extract_climate_features(valid)

    # Merge extracted features with original data
    train = pd.concat([train, train_features], axis=1)
    valid = pd.concat([valid, valid_features], axis=1)

    # Handle Missing Values
    train = handle_missing_values(train)
    valid = handle_missing_values(valid)

    # Save updated datasets
    train.to_csv(DATA_PATH + "Processed_Training_Data.csv", index=False)
    valid.to_csv(DATA_PATH + "Processed_Validation_Data.csv", index=False)

    print("📁 Final cleaned datasets saved successfully!")
    
    # ✅ Additional Check: Verify that no missing values remain
    print("\n🔍 Final check for missing values:")
    print(train.isnull().sum())
