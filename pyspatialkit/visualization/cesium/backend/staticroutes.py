from pathlib import Path
import os

from fastapi import APIRouter, Response, Request
from starlette.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

frontend_source_path = Path('/workspaces/pyspatialkit/frontend/spatialkitcesium/public')

index_html_path = Path(os.path.realpath(__file__)).parents[1] / 'frontend' / 'index.html'

router = APIRouter(
    tags=["static"],
    responses={404: {"description": "Not found (projects)"}},
)

@router.get("/static/")
async def get_index_html(request: Request, response_class=HTMLResponse):
    index_html = ''
    with open(frontend_source_path / "index.html") as file:
        index_html = file.read()
    return HTMLResponse(content=index_html, status_code=200)