#!/bin/bash

echo "Submit NB with CountVectorizer"
venv/bin/python multiple_models.py --run_name "naivebayes" --exp_name "sentiment_comparison" --model "naivebayes" --vectorizer "countvectorizer"

echo "Submit NB with TFIDF"
venv/bin/python multiple_models.py --run_name "naivebayes_tfidf" --exp_name "sentiment_comparison" --model "naivebayes" --vectorizer "tfidf"

echo "Submit LogReg with CountVectorizer"
venv/bin/python multiple_models.py --run_name "logreg" --exp_name "sentiment_comparison" --model "logreg" --vectorizer "countvectorizer"

echo "Submit LogReg with TFIDF"
venv/bin/python multiple_models.py --run_name "logreg_tfidf" --exp_name "sentiment_comparison" --model "logreg" --vectorizer "tfidf"

echo "Submit RF with CountVectorizer"
venv/bin/python multiple_models.py --run_name "randomforest" --exp_name "sentiment_comparison" --model "randomforest" --vectorizer "countvectorizer"

echo "Submit RF with TFIDF"
venv/bin/python multiple_models.py --run_name "randomforest_tfidf" --exp_name "sentiment_comparison" --model "randomforest" --vectorizer "tfidf"
