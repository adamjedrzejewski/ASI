"""
This is a boilerplate pipeline 'sentiment_analysis'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import transform_text_to_dataframe, concat_training_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(transform_text_to_dataframe, inputs="sentiment_analysis_training_data", outputs="sentiment_analysis_training_data_with_headers"),
        node(transform_text_to_dataframe, inputs="sentiment_analysis_test_data", outputs="sentiment_analysis_test_data_with_headers"),
        node(transform_text_to_dataframe, inputs="sentiment_analysis_val_data", outputs="sentiment_analysis_val_data_with_headers"),
        node(concat_training_data, inputs=["sentiment_analysis_training_data_with_headers", "sentiment_analysis_val_data_with_headers"], outputs="sentiment_analysis_model_training_data")
    ])
