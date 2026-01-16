
from fastapi import FastAPI
import os
from dotenv import load_dotenv

from rag.evaluation import query_test

load_dotenv()

app = FastAPI()


@app.get("/")
def healthcheck():
    return {"status": "ok"}
