"""
This is a boilerplate pipeline 'sprint_two_housing'
generated using Kedro 0.18.3
"""
import numpy as np
import pandas as pd

def add_synthetic_features(housing_raw_data: pd.DataFrame) -> pd.DataFrame:
    housing = housing_raw_data
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    return housing

