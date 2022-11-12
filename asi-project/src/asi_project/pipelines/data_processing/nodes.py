"""
This is a boilerplate pipeline 'sentiment_analysis'
generated using Kedro 0.18.3
"""

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import scipy
import re
import string
import logging

logger = logging.getLogger(__name__)

def transform_raw_text_to_dataframe(textContent: str) -> pd.DataFrame:
    text_list = []
    label_list = []
    for line in textContent.split('\n'):
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
    df.reset_index(inplace=True,drop=True)

    return df

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    dist_labels={}
    for num,key in enumerate(list(set(df.label))):
        dist_labels[key]=num 
    df['label']=df['label'].map(dist_labels)
    return df

def create_corpus(df: pd.DataFrame) -> pd.DataFrame:
    lm = WordNetLemmatizer()
    def clean_text(text):
        text = "".join([word.lower() for word in text if word not in string.punctuation])
        tokens = re.split('r\W+', text)
        text = [lm.lemmatize(word) for word in tokens if word not in set(stopwords.words('english'))]
        return ' '.join(str(x) for x in text)

    return df['text'].apply(lambda x:clean_text(x))

def extract_features(corpus: pd.DataFrame) -> scipy.sparse._csr.csr_matrix:
    tfidf_vect = TfidfVectorizer()
    return tfidf_vect.fit_transform(corpus)

def get_labels(df: pd.DataFrame) -> pd.core.series.Series:
    return df.label