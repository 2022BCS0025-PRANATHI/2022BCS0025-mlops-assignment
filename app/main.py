from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from typing import List

app = FastAPI()

ROLL_NO = "2022BCS0025"
NAME = "Mantri Pranathi"

# Load model
model = joblib.load("model.pkl")

# Input schema
class InputData(BaseModel):
    data: List[float]

# Health endpoint
@app.get("/")
def health():
    return {
        "name": NAME,
        "roll_no": ROLL_NO
    }

# Prediction endpoint
@app.post("/predict")
def predict(input: InputData):
    pred = model.predict([input.data])
    return {
        "prediction": int(pred[0]),
        "name": NAME,
        "roll_no": ROLL_NO
    }