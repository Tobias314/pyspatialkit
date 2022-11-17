from pathlib import Path
import os

from fastapi import APIRouter, Response, Request
from starlette.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .server import FRONTEND_PATH

router = APIRouter(
    tags=["static"],
    responses={404: {"description": "Not found (projects)"}},
)

@router.get("/static/")
async def get_index_html(request: Request, response_class=HTMLResponse):
    index_html = ''
    with open(FRONTEND_PATH / "index.html") as file:
        index_html = file.read()
    return HTMLResponse(content=index_html, status_code=200)