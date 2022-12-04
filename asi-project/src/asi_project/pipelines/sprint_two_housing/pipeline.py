"""
This is a boilerplate pipeline 'sprint_two_insurance'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import add_synthetic_features, find_best_model_with_pycaret, prepare_for_manual_training, train_model_and_store_evaluation_in_wandb, perform_hyperparameter_optimization_with_optuna


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            add_synthetic_features,
            inputs="housing_data_raw",
            outputs="housing_data_with_new_features"),
        node(
            find_best_model_with_pycaret,
            inputs="housing_data_with_new_features",
            outputs=["best_pycaret_model", "residuals_pycaret"]
        ),
        node(
            prepare_for_manual_training,
            inputs="housing_data_with_new_features",
            outputs=["housing_X_train", "housing_X_test", "housing_y_train", "housing_y_test"]
        ),
        node(
            train_model_and_store_evaluation_in_wandb,
            inputs=["housing_X_train", "housing_X_test", "housing_y_train", "housing_y_test"],
            outputs="housing_basic_rf_model"
        ),
        node(
            perform_hyperparameter_optimization_with_optuna,
            inputs=["housing_X_train", "housing_X_test", "housing_y_train", "housing_y_test"],
            outputs="optuna_results"
        )
    ])
