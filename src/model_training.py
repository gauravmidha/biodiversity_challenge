import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Define paths
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/')) + "/"
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/')) + "/"

# Ensure models directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

def load_data():
    """Load processed training data."""
    train = pd.read_csv(DATA_PATH + "Processed_Training_Data.csv")
    return train

def train_baseline_model():
    """Train a baseline Random Forest model."""
    print("📥 Loading processed data...")
    train = load_data()

    # Define feature columns (exclude latitude, longitude, and target variable)
    feature_columns = [col for col in train.columns if col not in ["Latitude", "Longitude", "Occurrence Status"]]
    
    # Define target variable
    X = train[feature_columns]
    y = train["Occurrence Status"]

    # Split data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🚀 Training Baseline Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, MODEL_PATH + "random_forest_baseline.pkl")
    print(f"💾 Baseline model saved to {MODEL_PATH + 'random_forest_baseline.pkl'}")

def tune_model():
    """Perform hyperparameter tuning on Random Forest model."""
    print("📥 Loading processed data for hyperparameter tuning...")
    train = load_data()

    # Define feature columns
    feature_columns = [col for col in train.columns if col not in ["Latitude", "Longitude", "Occurrence Status"]]
    
    # Define target variable
    X = train[feature_columns]
    y = train["Occurrence Status"]

    # Split data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🔍 Running hyperparameter tuning...")

    # Define parameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    # Perform Grid Search with cross-validation
    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best parameters
    best_params = grid_search.best_params_
    print(f"✅ Best Hyperparameters: {best_params}")

    # Train the final model with best parameters
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    # Save the best model
    joblib.dump(best_model, MODEL_PATH + "random_forest_tuned.pkl")
    print(f"💾 Optimized model saved to {MODEL_PATH + 'random_forest_tuned.pkl'}")
