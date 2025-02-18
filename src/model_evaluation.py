import os
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# Define paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/')) + "/"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/')) + "/"
REPORTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../reports/')) + "/"

# Ensure reports directory exists
os.makedirs(REPORTS_PATH, exist_ok=True)

def load_model(model_name):
    """Load a trained model from file."""
    print(f"📥 Loading trained model: {model_name}...")
    model = joblib.load(MODEL_PATH + model_name)
    return model

def load_test_data():
    """Load processed training data and split test set."""
    print("📥 Loading test dataset...")
    train = pd.read_csv(DATA_PATH + "Processed_Training_Data.csv")

    # Define feature columns
    feature_columns = [col for col in train.columns if col not in ["Latitude", "Longitude", "Occurrence Status"]]

    # Define target variable
    X = train[feature_columns]
    y = train["Occurrence Status"]

    # Use the last 20% of data as test set (same split from training)
    split_index = int(0.8 * len(train))
    X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

    return X_test, y_test

def evaluate_model(model_name):
    """Evaluate a trained model and save its metrics."""
    model = load_model(model_name)
    X_test, y_test = load_test_data()

    print(f"🔍 Evaluating model: {model_name}...")
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Save metrics
    report = {
        "model": model_name,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    report_file = REPORTS_PATH + model_name.replace(".pkl", "_evaluation.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)

    print(f"✅ Evaluation report saved: {report_file}")

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Frog", "Frog"], yticklabels=["No Frog", "Frog"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(REPORTS_PATH + model_name.replace(".pkl", "_confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    # Evaluate the baseline Random Forest model
    evaluate_model("random_forest_baseline.pkl")

    # Evaluate the optimized Random Forest model
    evaluate_model("random_forest_tuned.pkl")

    # Evaluate the XGBoost model
    evaluate_model("xgboost_model.pkl")

    print("📁 All model evaluation reports saved successfully!")

