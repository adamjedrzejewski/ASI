from enum import Enum
from fastapi import Body, FastAPI, Path
from pydantic import BaseModel

app = FastAPI()

class MyResult(BaseModel):
    hello: int

class Item(BaseModel):
    name: str
    cost: float

class MyOption(Enum):
    identity = "identity"
    negate = "negate"
    

@app.get("/")
async def main() -> MyResult:
    return {"hello": 3}

@app.get("/sum/{x}/{y}")
async def sum_(x: int, y: int, k: bool, j: str | None = None) -> MyResult:
    return MyResult(hello=x + y)

@app.delete("/sum1/{x}/{y}")
async def sum1(
    x: int,
    y: MyOption
) -> MyResult:
    return MyResult(hello=(x + 1) * (-1 if y == MyOption.negate else 1))

@app.post("/sum2/{x}/{y}")
async def sum2(
    x: int | None = Path(ge=0),
    y: MyOption | None = Path(),
    item: Item = Body()
) -> MyResult:
    print(item.name)
    return MyResult(hello=(x + 1) * (-1 if y == MyOption.negate else 1))