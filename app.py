from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pickle

# Load your machine learning model
current_directory = os.getcwd()
model_path = "./models/pest_disease.pickle"  # Replace with the path to your model
with open(model_path, 'rb') as model_file:
    pest_disease = pickle.load(model_file)

# Create a FastAPI app
app = FastAPI()

# Define input and output data models
class InputData(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    disease_num: int
    rain: int

class OutputData(BaseModel):
    status: str

# Function to map status_num to descriptive labels
def get_status_label(status_num):
    if status_num == 1:
        return "high"
    elif status_num == 2:
        return "medium"
    elif status_num == 3:
        return "low"
    else:
        return "unknown"
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

# Create a prediction endpoint for pest and disease model
@app.post("/pest-disease", response_model=OutputData)
async def predict_pest_disease(input_data: InputData):
    input_values = np.array([input_data.temperature, input_data.humidity, input_data.wind_speed, input_data.disease_num, input_data.rain]).reshape(1, -1)
    prediction = pest_disease.predict(input_values)
    status_label = get_status_label(int(prediction))
    return {"status": status_label}
