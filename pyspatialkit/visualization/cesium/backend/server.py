from pathlib import Path
from typing import List
import contextlib
import time
import threading
import argparse
import os
import webbrowser

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn


from ....storage.geostorage import GeoStorage
from .geostorageroutes import router as geostorage_router
from .rasterroutes import router as raster_router
from .staticroutes import router as static_router

frontend_path = Path(os.path.realpath(__file__)).parents[1] / 'frontend/'



class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

class LowerCaseMiddleware:
    def __init__(self) -> None:
        self.DECODE_FORMAT = "latin-1"

    async def __call__(self, request: Request, call_next):
        raw = request.scope["query_string"].decode(self.DECODE_FORMAT).upper()
        request.scope["query_string"] = raw.encode(self.DECODE_FORMAT)

        response = await call_next(request)
        return response

def start_server(geostorage: GeoStorage) -> None:
    print("STARTING SERVER")
    app = FastAPI()
    app.include_router(static_router)
    app.include_router(geostorage_router)
    app.include_router(raster_router)
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    my_middleware = LowerCaseMiddleware()
    app.middleware("http")(my_middleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.geostorage = geostorage
    config = uvicorn.Config(app, host="127.0.0.1", port=8080, log_level="info")
    server = Server(config=config)
    with server.run_in_thread():#TODO do not just wait for input here
        webbrowser.open("http://127.0.0.1:8080/")
        inp = input()

#print("TEST")
#app = FastAPI()
# app.geostorage = GeoStorage("./tests/tmp/geostorage/")
# app.include_router(static_router)
# app.include_router(geostorage_router)
# app.include_router(raster_router)
# app.mount("/static", StaticFiles(directory=frontend_path), name="static")
# my_middleware = LowerCaseMiddleware()
# app.middleware("http")(my_middleware)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )