"""
This is a boilerplate pipeline 'sprint_two_insurance'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import add_synthetic_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            add_synthetic_features,
            inputs="housing_data_raw",
            outputs="housing_data_with_new_features"),
    ])
