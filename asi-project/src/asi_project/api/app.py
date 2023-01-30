from fastapi import FastAPI
from pydantic import BaseModel

from typing import List

app = FastAPI()


class HousingDataPoint(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: int
    total_bedrooms: int
    population: int
    households: int
    median_income: int
    median_house_value: int
    ocean_proximity: str


@app.post("/housing")
async def housing(data: List[HousingDataPoint]):
    return data
