"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import transform_raw_text_to_dataframe, concat_training_data, encode_labels, create_corpus, \
    extract_features, get_labels, perform_grid_search


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            transform_raw_text_to_dataframe,
            inputs="training_data_raw",
            outputs="training_data_with_headers"),

        node(
            transform_raw_text_to_dataframe,
            inputs="test_data_raw",
            outputs="test_data_with_headers"),

        node(
            transform_raw_text_to_dataframe,
            inputs="val_data_raw",
            outputs="val_data_with_headers"),

        node(
            concat_training_data,
            inputs=["training_data_with_headers", "val_data_with_headers"],
            outputs="model_training_data"),

        node(
            encode_labels,
            inputs="model_training_data",
            outputs="training_data_encoded"
        ),

        node(
            create_corpus,
            inputs="training_data_encoded",
            outputs="text_corpus"
        ),

        node(
            extract_features,
            inputs="text_corpus",
            outputs="tfidf_features"
        ),

        node(
            get_labels,
            inputs="training_data_encoded",
            outputs="training_labels"
        ),

        node(
            perform_grid_search,
            inputs=["tfidf_features", "training_labels"],
            outputs="grid_search"
        )
    ])
