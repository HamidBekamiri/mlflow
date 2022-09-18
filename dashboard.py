import streamlit as st
import pandas as pd
import json
import boto3

global app_name
global region
app_name = 'NaiveBayesTest'
region = "eu-west-1"

def check_status(app_name):
    sage_client = boto3.client('sagemaker', region_name=region)
    endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
    endpoint_status = endpoint_description['EndpointStatus']
    return endpoint_status


def query_endpoint(app_name, input_json):
    client = boto3.session.Session().client('sagemaker-runtime', region)

    response = client.invoke_endpoint(
        EndpointName = app_name,
        Body = input_json,
        ContentType = 'application/json; format=pandas-split',
        )

    preds = response['Body'].read().decode('ascii')
    preds = json.loads(preds)
    print('Received response: {}'.format(preds))
    return preds


STYLES = {
    "candy": "candy",
    "composition 6": "composition_vii",
    "feathers": "feathers",
    "la_muse": "la_muse",
    "mosaic": "mosaic",
    "starry night": "starry_night",
    "the scream": "the_scream",
    "the wave": "the_wave",
    "udnie": "udnie",
}


# defines an h1 header
text = st.text_input("Insert tweet here")

input_df = pd.DataFrame({"text":text}, index=[0]).to_json(orient="split")

st.subheader("Sentiment:")

st.write( query_endpoint(app_name=app_name, input_json=input_df))
