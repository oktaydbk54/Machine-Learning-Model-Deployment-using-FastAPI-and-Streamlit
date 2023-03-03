from fastapi import FastAPI
from typing import List
import joblib

app = FastAPI()


@app.post('/predict/DecisionTree')
async def predict(data: List[float]):

    model = joblib.load("DecisionTree.pkl")

    prediction = model.predict([data])[0]
    return prediction

@app.post('/predict/KNN')
async def predict(data: List[float]):

    model = joblib.load("KNN.pkl")

    prediction = model.predict([data])[0]
    return prediction

@app.post('/predict/LogisticRegression')
async def predict(data: List[float]):

    model = joblib.load("logistic_regression_model.pkl")

    prediction = model.predict([data])[0]
    return prediction