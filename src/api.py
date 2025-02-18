import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Define paths
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/')) + "/"

# Load the trained model
model_filename = "random_forest_baseline.pkl"
model = joblib.load(MODEL_PATH + model_filename)

# Initialize FastAPI app
app = FastAPI(title="Biodiversity ML API", description="API for Predicting Frog Presence", version="1.0")

# Define request model
class PredictionRequest(BaseModel):
    climate_var_1: float
    climate_var_2: float

# Define response model
class PredictionResponse(BaseModel):
    prediction: int
    probability: float

@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "Welcome to the Biodiversity ML API! Use /predict to make predictions."}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    """Make predictions using the trained model."""
    input_data = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Get probability of positive class

    return {"prediction": int(prediction), "probability": float(probability)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)