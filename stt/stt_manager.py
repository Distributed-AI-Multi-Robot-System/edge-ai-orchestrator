import asyncio
import logging
import uuid
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import multiprocessing
import device_type as dt

# Importiere den Worker Code
from stt_worker import worker_logic

logger = logging.getLogger("API")



class STTManager:
    def __init__(self):
        self.ctx = multiprocessing.get_context('spawn')
        self.input_queue = self.ctx.Queue()
        self.output_queue = self.ctx.Queue()
        self.process = None
        self.listeners = {} # session_id -> asyncio.Queue

    def start(self):
        self.process = self.ctx.Process(
            target=worker_logic,
            args=(self.input_queue, self.output_queue, dt.DeviceType.CPU), # 'cpu' for mac testing
            daemon=True
        )
        self.process.start()
        # Start Output Reader Loop
        asyncio.create_task(self.broadcast_results())

    async def broadcast_results(self):
        """Holt Ergebnisse vom Prozess und verteilt sie an die Websockets"""
        loop = asyncio.get_running_loop()
        while True:
            try:
                # Non-blocking get from Multiprocessing Queue via Executor
                result = await loop.run_in_executor(None, self.output_queue.get)
                
                if result:
                    s_id = result.get("session_id")
                    if s_id in self.listeners:
                        # Push in die lokale Async Queue des spezifischen Websockets
                        await self.listeners[s_id].put(result)
            except Exception as e:
                logger.error(f"Broadcast Error: {e}")

    def register(self, session_id):
        self.listeners[session_id] = asyncio.Queue()
        return self.listeners[session_id]

    def unregister(self, session_id):
        if session_id in self.listeners:
            del self.listeners[session_id]
        self.input_queue.put({"type": "disconnect", "session_id": session_id})

    def stream_audio(self, session_id, audio_data):
        self.input_queue.put({
            "type": "audio",
            "session_id": session_id,
            "data": audio_data
        })