import os
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/')) + "/"
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/')) + "/"
REPORTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../reports/')) + "/"

# Ensure reports directory exists
os.makedirs(REPORTS_PATH, exist_ok=True)

def load_model(model_name):
    """Load a trained model from file."""
    print(f"📥 Loading trained model: {model_name}...")
    model = joblib.load(MODEL_PATH + model_name)
    return model

def extract_feature_importance(model_name):
    """Extract and visualize feature importance from a trained model."""
    model = load_model(model_name)

    # Load feature names
    train = pd.read_csv(DATA_PATH + "Processed_Training_Data.csv")
    feature_columns = [col for col in train.columns if col not in ["Latitude", "Longitude", "Occurrence Status"]]

    # Extract feature importance
    if hasattr(model, "feature_importances_"):  # Works for Random Forest & XGBoost
        feature_importance = model.feature_importances_
    else:
        raise ValueError(f"⚠ Model {model_name} does not support feature importance.")

    importance_df = pd.DataFrame({"Feature": feature_columns, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Save feature importance to a JSON file
    report_file = REPORTS_PATH + model_name.replace(".pkl", "_feature_importance.json")
    importance_df.to_json(report_file, orient="records", indent=4)
    print(f"✅ Feature importance report saved: {report_file}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(15), palette="viridis")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Feature Name")
    plt.title(f"Top 15 Most Important Features - {model_name}")
    plt.savefig(REPORTS_PATH + model_name.replace(".pkl", "_feature_importance.png"))
    plt.show()

if __name__ == "__main__":
    # Extract feature importance for Baseline Random Forest
    extract_feature_importance("random_forest_baseline.pkl")

    # Extract feature importance for Tuned Random Forest
    extract_feature_importance("random_forest_tuned.pkl")

    # Extract feature importance for XGBoost
    extract_feature_importance("xgboost_model.pkl")

    print("📊 Feature importance reports saved for all models!")
