"""
This is a boilerplate pipeline 'sentiment_analysis'
generated using Kedro 0.18.3
"""
import logging
import re
import string

import numpy as np
import pandas as pd
import scipy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


def transform_raw_text_to_dataframe(text_content: str) -> pd.DataFrame:
    text_list = []
    label_list = []
    for line in text_content.split('\n'):
        split = line.split(';')
        if len(split) != 2:
            continue

        text, label = split
        text_list.append(text)
        label_list.append(label)

    return pd.DataFrame({
        'text': text_list,
        'label': label_list
    })


def concat_training_data(training: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([training, val])
    df.reset_index(inplace=True, drop=True)

    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    dist_labels = {}
    for num, key in enumerate(list(set(df.label))):
        dist_labels[key] = num
    df['label'] = df['label'].map(dist_labels)
    return df


def create_corpus(df: pd.DataFrame) -> pd.DataFrame:
    lm = WordNetLemmatizer()

    def clean_text(text):
        text = "".join([word.lower() for word in text if word not in string.punctuation])
        tokens = re.split(r'\W+', text)
        text = [lm.lemmatize(word) for word in tokens if word not in set(stopwords.words('english'))]
        return ' '.join(str(x) for x in text)

    corpus = df['text'].apply(lambda x: clean_text(x))
    return corpus


def extract_features(corpus: pd.DataFrame) -> scipy.sparse._csr.csr_matrix:
    tfidf_vect = TfidfVectorizer()
    result = tfidf_vect.fit_transform(corpus)
    return result


def get_labels(df: pd.DataFrame) -> pd.core.series.Series:
    return df.label


def perform_grid_search(features: scipy.sparse._csr.csr_matrix, labels: pd.Series) -> GridSearchCV:
    parameters = {'max_features': ('auto', 'sqrt'),
                  'n_estimators': [500],
                  'max_depth': [5, None], }
    print(features.shape)
    grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=5, return_train_score=True, n_jobs=-1)
    grid_search.fit(features, np.array(labels).reshape(-1, 1))
    return grid_search.best_params_
