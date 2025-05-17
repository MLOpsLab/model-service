import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import uvicorn
import os

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

# Load latest Production model from MLflow Model Registry
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
model = mlflow.pyfunc.load_model(model_uri="models:/diabetes_rf_model@local")


app = FastAPI()

@app.post("/predict")
def predict(patients: List[Patient]):
    # Convert input to DataFrame
    df = pd.DataFrame([p.model_dump() for p in patients])
    preds = model.predict(df)
    return {"predictions": preds.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)