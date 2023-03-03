# Machine-Learning-Model-Deployment-using-FastAPI-and-Streamlit

This is a sample project demonstrating how to deploy machine learning models using FastAPI and Streamlit. The project contains a web interface built using Streamlit where users can input data and get predictions from different machine learning models. The predictions are made by a FastAPI app which serves the machine learning models as APIs.

# Project structure
The project has the following structure:

```├── app 
│   ├── api 
│   │   ├── model1.py 
│   │   ├── model2.py 
│   │   ├── model3.py 
│   ├── main.py 
├── models 
│   ├── model1.pkl 
│   ├── model2.pkl 
│   ├── model3.pkl 
├── streamlit_app 
│   ├── main.py 
├── README.md 
 ```

*`app/api`: Contains the FastAPI app and the code for each machine learning model API.\n

*`app/main.py`: Contains the FastAPI app startup and shutdown events.\n

*`models`: Contains the trained machine learning models in pickle format.

*`streamlit_app`: Contains the Streamlit web interface for the project.

*`.gitignore`: Specifies which files should be ignored by Git.

*`README.md`: This file.


# Usage

To run the project, first install the dependencies:

```pip install -r requirements.txt```

Then, start the FastAPI app:

```cd app
uvicorn main:app --reload
```

The API documentation can be accessed at http://localhost:8000/docs.

Finally, start the Streamlit app:

```streamlit run streamlit_app/main.py```


The web interface can be accessed at http://localhost:8501.

Contributing
Contributions are welcome! If you find any issues or want to add a new feature, please open an issue or submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.
