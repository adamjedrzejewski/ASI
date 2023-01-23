from fastapi import APIRouter
from kedro.runner import SequentialRunner
from kedro.io import DataCatalog
from classes.schemas.employee import Employee

router = APIRouter()

@router.put('/')
def main(employees: list[Employee],
    power: float) -> dict[str, str]:
    runner = SequentialRunner()
    catalog = DataCatalog(feed_dict={"xs": [e.salary for e in employees], "parameters": {"power" : power}})
    #runner.run(catalog=catalog)
    return {"Hello": "world!"}