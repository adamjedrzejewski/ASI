"""
This is a boilerplate pipeline 'sprint_two_housing'
generated using Kedro 0.18.3
"""
import numpy as np
import pandas as pd
import wandb
from pycaret.regression import setup

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

def add_synthetic_features(housing_raw_data: pd.DataFrame) -> pd.DataFrame:
    housing = housing_raw_data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    return housing

def find_best_model_with_pycaret(housing_data_with_new_features: pd.DataFrame):
    s = setup(housing_data_with_new_features, target='median_house_value')
    best_pycaret_model = compare_models()
    residuals_pycaret = plot_model(best, plot = 'residuals')
    return best_pycaret_model, residuals_pycaret

def prepare_for_manual_training(housing_data_with_new_features: pd.DataFrame):
    housing = housing_data_with_new_features
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_y_train = strat_train_test["median_house_value"].copy()
    housing_num = housing.drop('ocean_proximity', axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    housing_X_train = full_pipeline.fit_transform(housing)
    housing_X_test = strat_test_set.drop("median_house_value", axis=1)
    housing_y_test = strat_test_test["median_house_value"].copy()
    return housing_X_train, housing_X_test, housing_y_train, housing_y_test

def train_model_and_store_evaluation_in_wandb(housing_X_train, housing_X_test, housing_y_train, housing_y_test):
    wandb.init(project="asi-project-task", entity="asi-project")
    forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
    forest_reg.fit(housing_prepared, housing_labels)
    wandb.sklearn.plot_regressor(forest_reg, housing_X_train, housing_X_test, housing_y_train, housing_y_test, model_name="Random Forest")
    return housing_X_train, housing_X_test, housing_y_train, housing_y_test


def perform_hyperparameter_optimization_with_optuna:
