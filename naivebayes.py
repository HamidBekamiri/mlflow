from sqlite3 import DatabaseError
import pandas as pd
import nltk 
nltk.download("stopwords")
from nltk.corpus import stopwords 
import string 
import re
# tracker 
import mlflow 
from mlflow.tracking.client import MlflowClient
# vectorize words 
from sklearn.feature_extraction.text import CountVectorizer  
import os
import sys
# naive bayes 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import auc, roc_curve
# train test split 
from sklearn.model_selection import train_test_split

STOPWORDS = stopwords.words("english")

# HELPER FUNCTIONS 
def remove_stopwords(text):
    r""" Function to remove stopwords from tweets
    Parameters
    ----------
    text: str, input tweet 
    
    Return 
    ------
    str, cleared tweet
    """
    tweet_no_punct = [word for word in text.split() if not word in STOPWORDS]
    return ' '.join(tweet_no_punct)

def remove_punctuation(text):
    r""" Function to remove punctuation
    Parameters
    ----------
    text: str, input tweet 
    
    Return 
    ------
    str, cleared tweet"""
    outline = text.translate(str.maketrans('', '', string.punctuation))
    return outline

def remove_specific_chars(text):
    r""" Custom function to remove \n, \s+ or \' 
    Parameters
    ----------
    text: str, input tweet 
    
    Return 
    ------
    str, cleared tweet
    """
    # remove words after @
    outline = re.sub("@\S+ ", "", text) 
    # remove single quote
    outline = re.sub("\'", '', outline) 
    # remove new line
    outline = re.sub('\\n', '', outline)   
    return outline


tweets_df = pd.read_csv("split-data/X_train.csv")
target_df = pd.read_csv("split-data/y_train.csv")
# PREPROCESS
# drop the info we're not going to use 
# id, date, flag 
tweets_df.drop(columns=['ids', 'date', 'flag'], inplace=True)
# start the cleaning process 
# lower text 
tweets_df.loc[:,'lower_text'] = tweets_df['text'].str.lower() 
# remove stopwords 
tweets_df.loc[:,'clean_text1'] = tweets_df['lower_text'].apply(remove_stopwords)
# remove puncts 
tweets_df.loc[:,'clean_text2'] = tweets_df['clean_text1'].apply(remove_specific_chars)
# remove chars 
tweets_df.loc[:,'clean_text3'] = tweets_df['clean_text2'].apply(remove_punctuation)
print(tweets_df.head())

# create train, val and test
X_train, X_valid, y_train, y_valid = train_test_split(
        tweets_df['clean_text3'], target_df['sentiment'], train_size=0.75
    )

print(len(X_train), len(X_valid), len(y_train), len(y_valid))
# count vectorizer
vectorizer = CountVectorizer() 
# fit on the entire dataset 
input_data = vectorizer.fit(tweets_df['clean_text3'])
# transform 
X_train = vectorizer.transform(X_train)
X_valid = vectorizer.transform(X_valid)

# read the variable from the file 
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
# set up a client
mlflow_client = MlflowClient(tracking_uri=mlflow_tracking_uri)
# MODELLING 
classifier = MultinomialNB() 
model_name = "NaiveBayes"
run_name = "NB_model"
exp_name = "FirstSentimentTest"
try:
    print("setting up experiment ")
    experiment = mlflow.create_experiment(name = exp_name)
    experiment_id = experiment.experiment_id
except:
    experiment = mlflow_client.get_experiment_by_name(exp_name)
    experiment_id = experiment.experiment_id


print("Set up mlflow tracking uri")
mlflow.set_tracking_uri(mlflow_tracking_uri)
# context manager for basic logging 
# use mlflow for experiment tracking
with mlflow.start_run(experiment_id=experiment_id,
                    run_name=run_name,
                    nested=False,):
    print("Autlog")
    mlflow.sklearn.autolog(log_models=True,log_input_examples=True,log_model_signatures=True, )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_valid)
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    mlflow.sklearn.log_model(sk_model=classifier, artifact_path="model")


mlflow.end_run()