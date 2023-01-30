from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from kedro.runner import SequentialRunner
from kedro.io import DataCatalog

from typing import List, Union

from src.asi_project.pipelines.sprint_four_housing_fast.pipeline import create_pipeline

app = FastAPI()


class HousingDataPoint(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: Union[float, None]
    total_rooms: Union[float, None]
    total_bedrooms: Union[float, None]
    population: Union[float, None]
    households: Union[float, None]
    median_income: Union[float, None]
    median_house_value: Union[float, None]
    ocean_proximity: str


@app.post("/housing")
async def housing(data: List[HousingDataPoint]):
    runner = SequentialRunner()
    results = []
    catalog = DataCatalog(feed_dict={"dataset": data, "results": results})
    runner.run(pipeline=create_pipeline(),catalog=catalog)
    return catalog.load("results")
