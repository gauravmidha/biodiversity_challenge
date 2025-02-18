import os
import pandas as pd
import joblib  # Load trained model

# Define paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/')) + "/"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/frog_presence_model.pkl'))

def load_model():
    """Load the trained Random Forest model."""
    print("📥 Loading trained model...")
    model = joblib.load(MODEL_PATH)
    return model

def load_validation_data():
    """Load processed validation dataset."""
    print("📥 Loading validation dataset...")
    validation_data = pd.read_csv(DATA_PATH + "Processed_Validation_Data.csv")
    return validation_data

def make_predictions():
    """Make predictions on the validation dataset."""
    model = load_model()
    validation_data = load_validation_data()

    # Select feature columns (excluding Latitude, Longitude)
    feature_columns = [col for col in validation_data.columns if col not in ["Latitude", "Longitude"]]

    # Predict frog presence
    print("🔍 Making predictions on validation data...")
    predictions = model.predict(validation_data[feature_columns])

    # Save predictions
    validation_data["Predicted_Occurrence"] = predictions
    validation_data.to_csv(DATA_PATH + "Predicted_Validation_Data.csv", index=False)

    print("✅ Predictions saved to data/Predicted_Validation_Data.csv")

if __name__ == "__main__":
    make_predictions()