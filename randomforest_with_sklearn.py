import pandas as pd
import nltk 
nltk.download("stopwords")
from nltk.corpus import stopwords 
import string 
import re
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
from sklearn.feature_extraction.text import TfidfVectorizer  
import os
import sys
# random forest
from sklearn.ensemble import RandomForestClassifier
# train test split 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
# numpy 
import numpy as np 
# matplot lib 
import matplotlib.pyplot as plt
# logging
import logging 
logger = logging.getLogger(__name__)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm
import gc
from contextlib import contextmanager
from sklearn.base import BaseEstimator, TransformerMixin



STOPWORDS = stopwords.words("english")


class PreprocessTweets(BaseEstimator, TransformerMixin):
    r""" This class implement the entire preprocess operation on the 
    input tweets
    """
    def __init__(self, feature_name = "text"):
        self.feature_name = feature_name 
    
    def fit(self, x, y=None):
        return self 
    
    def clean_data(self, x):
        r""" Main cleaning function
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
        X = X.copy() 
        X.loc[:,'text'] = X[self.feature_name].apply(lambda x: self.clean_data(x))
        #output_array = X['cleaned_text'].to_numpy()
        return X['text'].to_numpy()


def get_model():
    r""" this function can be further personalized with a given 
    input for th especific model we want
    """
    classifier = RandomForestClassifier(n_estimators=50, random_state=0)
    return classifier 


def training_process():
    r""" Full function to run the preprocess and training 
    """
    # retrieve  the model 
    classifier = get_model()
    # create the pipeline
    training_pipeline = Pipeline(steps=[
        ("clean", PreprocessTweets("text")), 
        ("countVectorizer", TfidfVectorizer(ngram_range=(1,4),
                                            use_idf=True,
                                            smooth_idf=True,
                                            sublinear_tf=True,
                                            analyzer='word',
                                            token_pattern=r'\w{1,}',
                                            max_features=1000)
                                            ), 
        ("trainModel", classifier)
        ]
    )
    return training_pipeline

def fitwarning():
    # This is not a lambda, because lambdas are not pickleable
    logger.warning(
        "Model pipeline should not be refitted at this stage, but confined to .transform and .predict* methods"
    )
    return False

tweets_df = pd.read_csv("split-data/X_train.csv")
target_df = pd.read_csv("split-data/y_train.csv")
# PREPROCESS

# drop the info we're not going to use  id, date, flag 
tweets_df.drop(columns=['user','ids', 'date', 'flag'], inplace=True)
tweets_df.reset_index(drop=True, inplace=True)
# train/test split
X_train, X_valid, y_train, y_valid = train_test_split(
        tweets_df, target_df['sentiment'], train_size=0.75
    )
print("splits")
print(X_train)
print(X_valid)
print(y_train)
print(y_valid)
print("Shape")
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
# TOOD SET UP EXPERIMETNS AND DO NOT USE THE CONTEXT MANAGER
mlflow_client = MlflowClient(tracking_uri=mlflow_tracking_uri)
model_name = "RandomForest"
run_name = "RF_model"
experiment_family = "RandomForest"
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
                           nested=False,) # we can add tags
print('artifact uri:', mlflow.get_artifact_uri())
# set the autolog 
mlflow.sklearn.autolog(log_models=True,log_input_examples=True,log_model_signatures=True, )
trained_model = training_process()
trained_model.fit(X_train, y_train)

y_pred = trained_model.predict(X_valid)
report = classification_report(
        y_valid, y_pred, output_dict=True
    )
cm = confusion_matrix(y_valid, y_pred)
joblib.dump(trained_model, "final_model.joblib")
mlflow.sklearn.log_model(sk_model=trained_model, artifact_path="model")
mlflow.end_run()