from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
import logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load STT worker + TTS worker ...
    stt_worker_dummy_var = "STT Worker Initialized"
    
    yield
    # Clean up the ML models and release the resources
    stt_worker_dummy_close_var = "STT Worker Closed"


app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")