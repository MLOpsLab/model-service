import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import uvicorn
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define your Patient model (for Pydantic validation)
class Patient(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Get environment variables
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MODEL_NAME = os.environ.get("MODEL_NAME", "diabetes_model-training")
RUN_ID = os.environ.get("RUN_ID", "f0aac2b2d54e4aafa3d8ea9dcd9de055")  # From your logs

# Determine model URI based on available information
if RUN_ID:
    MODEL_URI = f"runs:/{RUN_ID}/model"
    logger.info(f"Using model from run: {MODEL_URI}")
elif MODEL_NAME:
    MODEL_URI = f"models:/{MODEL_NAME}/latest"
    logger.info(f"Using latest model from registry: {MODEL_URI}")
else:
    raise ValueError("Either RUN_ID or MODEL_NAME must be provided")

# Create FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts diabetes using MLflow model",
    version="1.0.0"
)

# Load model during startup
@app.on_event("startup")
async def load_model():
    global model
    logger.info(f"Setting MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    logger.info(f"Loading model from: {MODEL_URI}")
    try:
        model = mlflow.pyfunc.load_model(model_uri=MODEL_URI)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Let the application start, but predictions will fail

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if "model" in globals() else "unhealthy",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri": MODEL_URI
    }

@app.post("/predict")
def predict(patients: List[Patient]):
    """
    Make diabetes predictions for a list of patients

    Returns:
        dict: Predictions (0 = non-diabetic, 1 = diabetic)
    """
    if "model" not in globals():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([p.model_dump() for p in patients])
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
