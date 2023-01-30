"""
This is a boilerplate pipeline 'sprint_four_housing_fast'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import add_synthetic_features, prepare_for_manual_training, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            add_synthetic_features,
            inputs="dataset",
            outputs="housing"),
        node(
            prepare_for_manual_training,
            inputs="housing",
            outputs=["housing_X_train", "housing_X_test", "housing_y_train", "housing_y_test"]
        ),
        node(
            train_model,
            inputs=["housing_X_train", "housing_X_test", "housing_y_train", "housing_y_test"],
            outputs="results"
        )
    ])
