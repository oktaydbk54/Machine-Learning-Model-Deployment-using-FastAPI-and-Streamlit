import streamlit as st
import requests
import json

# FastAPI endpoint
API_URL = "http://localhost:8000/predict"

# Model options dictionary
models = {
    "DecisionTree": "DecisionTree",
    "KNN": "KNN",
    "LogisticRegression": "logistic_regression_model"
}

# Streamlit app
def main():
    st.title("Machine Learning Model Predictor")

    # Model selection dropdown
    selected_model = st.selectbox("Select a model", list(models.keys()))

    # Get model file name based on selection
    model_file = models[selected_model]

    # Feature inputs
    sepal_length = st.number_input("Sepal length")
    sepal_width = st.number_input("Sepal width")
    petal_length = st.number_input("Petal length")
    petal_width = st.number_input("Petal width")

    # Make prediction on button click
    if st.button("Predict"):
        # Prepare feature data as JSON payload
        feature_data = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        feature_data = [sepal_length, sepal_width, petal_length, petal_width]

        # Call FastAPI endpoint and get prediction result
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_URL + f"/{model_file}", json=feature_data)

        # Display prediction result
        st.write(f"Prediction: {response.json()}")
    
if __name__ == "__main__":
    main()
