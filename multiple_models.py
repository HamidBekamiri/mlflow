import pandas as pd
import nltk 
nltk.download("stopwords")
from nltk.corpus import stopwords 
import string 
import re
import os
import sys
# save model 
import joblib
# tracker 
#from dagshub import dagshub_logger, DAGsHubLogger 
import mlflow 
from mlflow.tracking.client import MlflowClient
# base class 
from sklearn.base import BaseEstimator, TransformerMixin
# pipeline 
from sklearn.pipeline import Pipeline
# vectorize words 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  
# random forest
from sklearn.ensemble import RandomForestClassifier
# log reg 
from sklearn.linear_model import LogisticRegression 
# NB
from sklearn.naive_bayes import MultinomialNB 
# train test split 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# numpy 
import numpy as np 
# matplot lib 
import matplotlib.pyplot as plt
# argparse
import argparse
# logging
import logging 
logger = logging.getLogger(__name__)


STOPWORDS = stopwords.words("english")


class PreprocessTweets(BaseEstimator, TransformerMixin):
    r""" This class implements the full cleaning process for the input text
    """
    def __init__(self, feature_name = "text"):
        r""" Constructor
        Parameters
        ----------
        feature_name: str, name of the feature to be taken from the input dataframe"""
        self.feature_name = feature_name 
    
    def fit(self, x, y=None):
        r""" Fit method"""
        return self 
    
    def clean_data(self, x):
        r""" Main cleaning function. 
        Here we are:
        - lowering text
        - remove stopwords 
        - tags and hashtags 
        - single quotes
        - new lines
        - punctuation
        """
        # lower text 
        x = x.lower()
        # remove stopwords
        x = [word for word in x.split() if not word in STOPWORDS]
        x = ' '.join(x)
        # remove words after @
        x = re.sub("@\S+ ", "", x) 
        # remove single quote
        x = re.sub("\'", '', x) 
        # remove new line
        x = re.sub('\\n', '', x)   
        # remove puncts 
        x = x.translate(str.maketrans('', '', string.punctuation))

        return x 
    
    def transform(self, X):
        r""" Transform function which maps clean_data out of X"""
        return self.fit_transform(X) 
    
    def fit_transform(self, X, y=0):
        r""" Fit transform method to clean the data
        Return 
        ------
        array: np.array, cleaned numpy array with text"""
        X = X.copy() 
        X.loc[:,'text'] = X[self.feature_name].apply(lambda x: self.clean_data(x))
        return X['text'].to_numpy()


def get_model(model:str):
    r""" This function return a specific model given the input string 
    Possible model: `naivebayes`, `logreg` or `randomforest` 

    Parameters
    ----------
    model: str, input type of model we want 

    Return 
    ------
    classifier: sklearn model
    """
    if model=="naivebayes":
        print(f"selected model Naive Bayes")
        classifier = MultinomialNB() 
    elif model=="logreg":
        print(f"selected model Logistic Regression")
        classifier = LogisticRegression(C=1, solver='sag') 
    elif model=="randomforest":
        print(f"selected model Random Forest")
        classifier = RandomForestClassifier(n_estimators=50, random_state=0)
    else:
        print(f"Model {model} doesn't exist. Please select among:")
        print("naivebayes, logreg and randomforest")
        sys.exit(-1)
    return classifier 


def training_process(model:str, 
                     vectorizer:str):
    r""" Function to create the training pipeline with cleaner and model 

    Parameters
    ----------
    model: str, type of model we want to run, see get_model function 
    vectorizer: str, type of vectorizer, `countvectorizer` or `tfidf`

    Return 
    -------
    training_pipeline: sklearn.pipeline with data cleaner and model
    """
    # retrieve  the model 
    classifier = get_model(model)
    if vectorizer=="countvectorizer":
        print(f"selected vectorizer CountVectorizer")
        vector = CountVectorizer()
    elif vectorizer=="tfidf":
        print(f"selected vectorizer TfidfVectorizer")
        vector = TfidfVectorizer(ngram_range=(1,4),
                                use_idf=True,
                                smooth_idf=True,
                                sublinear_tf=True,
                                analyzer='word',
                                token_pattern=r'\w{1,}',
                                max_features=1000)
    else:
        print(f"Vectorizer {vectorizer} doesn't exist. Please select among:")
        print("countvectorizer or tfidf")
        sys.exit(-1)

    # create the pipeline
    training_pipeline = Pipeline(steps=[
        ("clean", PreprocessTweets("text")), 
        ("countVectorizer",vector), 
        ("trainModel", classifier)
        ]
    )
    return training_pipeline

# MAIN
parser = argparse.ArgumentParser(description='Input arguments for MLflow testing in general')
parser.add_argument('--run_name', type=str, 
                    help='Name of the run within the experiment family')
parser.add_argument('--exp_name', type=str,
                    help='Name of the family experiment')
parser.add_argument('--model', type=str,
                    help='Model to be used: naivebayes, logreg, randomforest')
parser.add_argument('--vectorizer', type=str,
                    help='Vectorizer to process text: countvectorizer or tfidf')

args = parser.parse_args()
run_name_ = args.run_name 
exp_name_ = args.exp_name 
model_ = args.model 
vectorizer_ = args.vectorizer 

tweets_df = pd.read_csv("split-data/X_train.csv")
target_df = pd.read_csv("split-data/y_train.csv")
# preprocessing
# drop the info we're not going to use  id, date, flag 
tweets_df.drop(columns=['user','ids', 'date', 'flag'], inplace=True)
tweets_df.reset_index(drop=True, inplace=True)
# train/test split
X_train, X_valid, y_train, y_valid = train_test_split(
        tweets_df, target_df['sentiment'], train_size=0.75
    )
print(len(X_train), len(X_valid), len(y_train), len(y_valid))
# setup MLflow
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

print("Set up mlflow tracking uri")

mlflow_client = MlflowClient(tracking_uri=mlflow_tracking_uri)
run_name = run_name_
experiment_family = exp_name_
try:
    print("setting up experiment ")
    experiment = mlflow.create_experiment(name = experiment_family)
    experiment_id = experiment.experiment_id
except:
    experiment = mlflow_client.get_experiment_by_name(experiment_family)
    experiment_id = experiment.experiment_id

print(f"Setting up experiment {experiment_family}")#
print(f"Experiment id {experiment_id}")
print(f"Run name {run_name}")
mlflow.set_tracking_uri(mlflow_tracking_uri)
# start the recording 
starter = mlflow.start_run(experiment_id=experiment_id,
                           run_name=run_name,
                           nested=False) 
print('artifact uri:', mlflow.get_artifact_uri())
# set the autolog 
mlflow.sklearn.autolog(log_models=True,log_input_examples=True,log_model_signatures=True, )
trained_model = training_process(model_, vectorizer_)
trained_model.fit(X_train, y_train)

y_pred = trained_model.predict(X_valid)
report = classification_report(
        y_valid, y_pred, output_dict=True
    )
cm = confusion_matrix(y_valid, y_pred)
joblib.dump(trained_model, "final_model.joblib")
mlflow.sklearn.log_model(sk_model=trained_model, artifact_path="model")
mlflow.end_run()