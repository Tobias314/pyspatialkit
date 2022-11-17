from fastapi import FastAPI, Depends
from pydantic import BaseModel

class TestModel(BaseModel):
    test: int
    lol: str

class FooModel(BaseModel):
    bar: int

app = FastAPI()


@app.get("/{test}")
async def root(model: TestModel = Depends(), fooModel: FooModel = Depends()):
    return {"message": model.test}