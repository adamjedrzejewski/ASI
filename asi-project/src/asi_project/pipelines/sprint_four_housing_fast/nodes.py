import numpy as np
import pandas as pd

"""
This is a boilerplate pipeline 'sprint_four_housing_fast'
generated using Kedro 0.18.3
"""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


def add_synthetic_features(dataset) -> pd.DataFrame:
    housing = pd.DataFrame([o.__dict__ for o in dataset])
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    return housing


def prepare_for_manual_training(housing: pd.DataFrame):
    housing = housing
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_test = strat_test_set.drop("median_house_value", axis=1)
    housing_num = housing.drop('ocean_proximity', axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    housing_X_train = full_pipeline.fit_transform(housing)
    housing_y_train = strat_train_set["median_house_value"].copy()
    housing_X_test = full_pipeline.fit_transform(housing_test)
    housing_y_test = strat_test_set["median_house_value"].copy()
    return housing_X_train, housing_X_test, housing_y_train, housing_y_test


def train_model(housing_X_train, housing_X_test, housing_y_train, housing_y_test,
                                              ):
    forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
    forest_reg.fit(housing_X_train, housing_y_train)
    results = forest_reg.predict(housing_X_test)
    return results.tolist()
