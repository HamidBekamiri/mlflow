# build up the backend of our model 
import os
from fastapi import FastAPI, Request
from fastpi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import mlflow 
import pandas as pd

app = FastAPI() 
# input variables
ifile = open("setup_mlflow.txt", "r").readlines()
mlflow_tracking_uri = ifile[0].split("=")[1].strip()
mlflow_tracking_username = ifile[1].split("=")[1].strip()
mlflow_tracking_password = ifile[2].split("=")[1].strip()
os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_tracking_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_tracking_password
print(os.environ.get("MLFLOW_TRACKING_URI"))
print(os.environ.get("MLFLOW_TRACKING_USERNAME"))
print(os.environ.get("MLFLOW_TRACKING_PASSWORD"))
print(os.environ)
mlflow.set_tracking_uri(mlflow_tracking_uri)
# load the model
logged_model = 'runs:/0beeaa491f144dffb3950e93980e4c20/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

@app.post("/predict")
async def predict():
    r""" Give a description"""
    test_df = pd.read_csv()
    preds = loaded_model.predict(test_df)

    json_data = jsonable_encoder(preds)
    return JSONResponse(content=json_data)
