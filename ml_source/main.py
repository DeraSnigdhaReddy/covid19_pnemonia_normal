from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "covid_diag.pkl")
model = joblib.load(model_path)


class inp(BaseModel):
    Age:int
    Gender:int
    Fever:int
    Cough:int
    Fatigue:int
    Breathlessness:int
    Comorbidity:int
    Stage:int
    Type:int
    Tumor_Size:float

app=FastAPI()

@app.get("/")
def route():
    return {"message":"Welcome"}

@app.post("/prd")
def prediction(data:inp):
    inp=pd.DataFrame([data.dict()])
    pred=model.predict(inp)
    return {"Prediction":pred[0]}

# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI()

# class PatientData(BaseModel):
#     Age: int
#     Gender: int
#     Fever: int
#     Cough: int
#     Fatigue: int
#     Breathlessness: int
#     Comorbidity: int
#     Stage: int
#     Type: int
#     Tumor_Size: int

# @app.post("/prd")
# def predict(data: PatientData):
#     # Dummy logic; replace with actual model prediction
#     score = (data.Age + data.Fever + data.Cough + data.Tumor_Size) / 20
#     return {"Prediction": round(score, 2)}
