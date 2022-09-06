import pandas as pd
import nltk 
nltk.download("stopwords")
from nltk.corpus import stopwords 
import string 
import re
# tracker 
from dagshub import dagshub_logger, DAGsHubLogger 

# vectorize words 
from sklearn.feature_extraction.text import CountVectorizer  

# naive bayes 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import auc, roc_curve, ConfusionMatrixDisplay

STOPWORDS = stopwords.words("english")

# PREPROCESSING 
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

vectorizer = CountVectorizer() 
X_train = vectorizer.fit_transform(tweets_df['clean_text3'])

# MODELLING 

# context manager 
with dagshub_logger() as logger: 
    classifier = MultinomialNB() 
    model_name = "NaiveBayes"
    run_name = "NB_model"
    # log model's parameters 
    logger.log_hyperparams(model_class=type(classifier).__name__)
    logger.log_hyperparams({"NaiveBayes": classifier.get_params()})
    classifier.fit(X_train, target_df.values.ravel())
    y_pred = classifier.predict_proba(X_train)[:,1]
    fpr, tpr, thresholds = roc_curve(target_df.values.ravel(), y_pred)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    logger.log_metrics({f'Training AUC':roc_auc})

    #ConfusionMatrixDisplay.from_estimator(classifier, X_train, target_df.values.ravel(), 
    #                                     display_labels=["1", "0"],
    #                                     )
    
    #logger.log_artifact()
