"""
This is a boilerplate pipeline 'sprint_two_housing'
generated using Kedro 0.18.3
"""
import numpy as np
import pandas as pd
import wandb
from pycaret.regression import setup, compare_models, plot_model

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import optuna

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
    residuals_pycaret = plot_model(best_pycaret_model, plot = 'residuals')
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
    return forest_reg


def perform_hyperparameter_optimization_with_optuna(housing_X_train, housing_X_test, housing_y_train, housing_y_test):
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]  # Create the random grid

    def objective(trial):
        max_features = trial.suggest_categorical("max_features", ['log2', 'sqrt', 1.0])
        min_samples_split = trial.suggest_categorical("min_samples_split",  [2, 5, 10])
        min_samples_leaf = trial.suggest_categorical("min_samples_leaf", min_samples_leaf = [1, 2, 4])
        max_depth = trial.suggest_categorical("max_depth", [None, 10, 21, 32, 43, 54])
        n_estimators = trial.suggest_int("n_estimators", low=100, max=1000, step=100)
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=42)
        rf.fit(housing_X_train, housing_y_train)
        prediction = rf.predict(housing_X_test)
        mse = mean_squared_error(y_test, prediction)
        return mse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    optuna_results = study.best_trial
    return optuna_results


