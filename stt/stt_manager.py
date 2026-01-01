import asyncio
import logging
import os
import multiprocessing

from stt.stt_worker import worker_logic
from stt.device_type import DeviceType

logger = logging.getLogger(__name__)


class STTManager:
    """Manages the STT worker process and session-based result distribution."""
    
    def __init__(self):
        self.ctx = multiprocessing.get_context('spawn')
        self.input_queue = self.ctx.Queue()
        self.output_queue = self.ctx.Queue()
        self.process = None
        self.listeners: dict[str, asyncio.Queue] = {}
        self._running = False
        self._broadcast_task = None
        
        # Load configuration from environment
        device_str = os.getenv("STT_DEVICE", "cpu").lower()
        self.device_type = DeviceType.CUDA if device_str == "cuda" else DeviceType.CPU
        self.model_bg = os.getenv("STT_MODEL_BG", "base")
        self.model_tail = os.getenv("STT_MODEL_TAIL", "tiny")
        
        logger.info(f"STTManager config: device={self.device_type.value}, bg_model={self.model_bg}, tail_model={self.model_tail}")

    def start(self):
        """Start the STT worker process and result broadcaster."""
        self.process = self.ctx.Process(
            target=worker_logic,
            args=(self.input_queue, self.output_queue, self.device_type, self.model_bg, self.model_tail),
            daemon=True
        )
        self.process.start()
        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_results())
        logger.info("STT worker process started")

    async def _broadcast_results(self):
        """Fetch results from worker process and distribute to session listeners."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                # Non-blocking get from multiprocessing queue via executor
                result = await loop.run_in_executor(None, self.output_queue.get)
                
                if result is None:
                    # Poison pill received, exit loop
                    break
                
                session_id = result.get("session_id")
                if session_id in self.listeners:
                    await self.listeners[session_id].put(result)
                    
            except Exception as e:
                if self._running:
                    logger.error(f"Broadcast error: {e}")

    def stop(self):
        """Gracefully stop the STT worker process."""
        self._running = False
        
        # Send poison pill to worker
        self.input_queue.put(None)
        
        # Cancel broadcast task
        if self._broadcast_task:
            self._broadcast_task.cancel()
        
        # Terminate process if still running
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            
        logger.info("STT worker process stopped")

    def register(self, session_id: str) -> asyncio.Queue:
        """Register a session and return its result queue."""
        self.listeners[session_id] = asyncio.Queue()
        logger.debug(f"Session {session_id} registered")
        return self.listeners[session_id]

    def unregister(self, session_id: str):
        """Unregister a session and notify worker to clean up."""
        if session_id in self.listeners:
            del self.listeners[session_id]
        self.input_queue.put({"type": "disconnect", "session_id": session_id})
        logger.debug(f"Session {session_id} unregistered")

    def stream_audio(self, session_id: str, audio_bytes: bytes):
        """Send raw audio bytes to the worker for processing."""
        self.input_queue.put({
            "type": "audio",
            "session_id": session_id,
            "data": audio_bytes
        })