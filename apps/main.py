from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

# Project folder ka path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "predictive_maintenance_model.pkl"

# Trained ML model load karna
model = joblib.load(MODEL_PATH)

# FastAPI application
app = FastAPI(
    title="Intelligent Predictive Maintenance System",
    description="ML-based machine failure prediction API",
    version="1.0.0"
)


# Input data ka format
class MachineData(BaseModel):
    Type: Literal["L", "M", "H"]
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


# Home route
@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running!"
    }


# Machine failure prediction route
@app.post("/predict")
def predict(data: MachineData):

    # User input ko DataFrame mein convert karna
    input_data = pd.DataFrame([{
        "Type": data.Type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear
    }])

    # Categorical Type ko numerical columns mein convert karna
    input_encoded = pd.get_dummies(
        input_data,
        columns=["Type"],
        dtype=int
    )

    # Training wale features aur unka order maintain karna
    input_encoded = input_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Prediction
    prediction = int(model.predict(input_encoded)[0])

    # Prediction probabilities
    probabilities = model.predict_proba(input_encoded)[0]

    # Recommendation
    if prediction == 1:
        recommendation = (
            "Potential failure detected. "
            "Inspect the machine and check its sensor readings."
        )
    else:
        recommendation = (
            "No failure predicted. "
            "Continue routine monitoring and scheduled maintenance."
        )

    # API response
    return {
        "machine_failure_prediction": prediction,
        "probability_no_failure": round(float(probabilities[0]), 4),
        "probability_failure": round(float(probabilities[1]), 4),
        "recommendation": recommendation
    }
