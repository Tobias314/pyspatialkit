from pathlib import Path
import os

from fastapi import APIRouter, Response, Request
from starlette.responses import HTMLResponse

index_html_path = Path(os.path.realpath(__file__)).parents[1] / 'frontend' / 'index.html'

router = APIRouter(
    tags=["static"],
    responses={404: {"description": "Not found (projects)"}},
)

@router.get("/")
async def get_index_html(request: Request, response_class=HTMLResponse):
    index_html = ''
    with open(index_html_path) as file:
        index_html = file.read()
    return HTMLResponse(content=index_html, status_code=200)
