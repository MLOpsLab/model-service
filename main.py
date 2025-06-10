import mlflow.pyfunc
import boto3
import joblib
from io import BytesIO
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

# Configuration from environment variables
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MODEL_NAME = os.environ.get("MODEL_NAME")
RUN_ID = os.environ.get("RUN_ID", "f0aac2b2d54e4aafa3d8ea9dcd9de055")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_KEY = os.environ.get("S3_KEY", "models/diabetes_model.joblib")

app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts diabetes using model from MLflow or S3",
    version="1.0.0"
)

# Function to load model from S3
def load_model_from_s3(bucket, key):
    logger.info(f"Loading model from S3: s3://{bucket}/{key}")
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        model_data = response['Body'].read()
        model = joblib.load(BytesIO(model_data))
        logger.info("Model loaded successfully from S3")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from S3: {e}")
        raise

# Function to load model from MLflow
def load_model_from_mlflow(uri):
    logger.info(f"Loading model from MLflow: {uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri=uri)
        logger.info("Model loaded successfully from MLflow")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        raise

# Load model at startup
@app.on_event("startup")
async def startup_event():
    global model, model_source

    # Try MLflow first if configured
    if MLFLOW_TRACKING_URI and (MODEL_NAME or RUN_ID):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            if RUN_ID:
                model_uri = f"runs:/{RUN_ID}/model"
            else:
                model_uri = f"models:/{MODEL_NAME}/latest"

            model = load_model_from_mlflow(model_uri)
            model_source = "mlflow"
            return
        except Exception as e:
            logger.warning(f"MLflow loading failed, falling back to S3: {e}")

    # Fall back to S3 if MLflow fails or isn't configured
    if S3_BUCKET and S3_KEY:
        try:
            model = load_model_from_s3(S3_BUCKET, S3_KEY)
            model_source = "s3"
        except Exception as e:
            logger.error(f"S3 loading failed: {e}")
            model_source = "none"
    else:
        logger.error("Neither MLflow nor S3 is properly configured")
        model_source = "none"

@app.get("/health")
def health_check():
    """Health check endpoint"""
    is_healthy = hasattr(globals(), 'model') and 'model' in globals()
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_source": model_source if is_healthy else "none",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "s3_location": f"s3://{S3_BUCKET}/{S3_KEY}" if S3_BUCKET and S3_KEY else None
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