import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Define paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/')) + "/"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/')) + "/"

# Ensure models directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

def load_data():
    """Load processed training data with selected important features."""
    train = pd.read_csv(DATA_PATH + "Processed_Training_Data.csv")

    # Keep only the most important features
    selected_features = ["climate_var_1", "climate_var_2", "Occurrence Status"]
    train = train[selected_features]

    return train

def train_xgboost():
    """Train an XGBoost model using only important features."""
    print("📥 Loading processed data for XGBoost...")
    train = load_data()

    # Define feature columns and target variable
    X = train[["climate_var_1", "climate_var_2"]]
    y = train["Occurrence Status"]

    # Split data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🚀 Training XGBoost model...")
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Save the trained XGBoost model
    joblib.dump(model, MODEL_PATH + "xgboost_model.pkl")
    print(f"💾 XGBoost model saved to {MODEL_PATH + 'xgboost_model.pkl'}")