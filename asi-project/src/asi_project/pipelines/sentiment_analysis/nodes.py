"""
This is a boilerplate pipeline 'sentiment_analysis'
generated using Kedro 0.18.3
"""

import pandas as pd

import logging

logger = logging.getLogger(__name__)

def transform_text_to_dataframe(textContent: str) -> pd.DataFrame:
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