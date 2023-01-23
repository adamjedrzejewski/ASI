from fastapi import FastAPI
from classes.schemas.employee import Employee
from classes.routers.data_processing import data_processing

app = FastAPI()

app.include_router(data_processing.router)