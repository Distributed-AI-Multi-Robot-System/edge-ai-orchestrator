import asyncio
import uuid
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from stt import STTManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global manager instances
stt_manager: STTManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global stt_manager
    
    # Startup: Initialize STT worker
    logger.info("Starting STT manager...")
    stt_manager = STTManager()
    stt_manager.start()
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("Stopping STT manager...")
    if stt_manager:
        stt_manager.stop()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/stt")
async def stt_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Speech-to-Text.
    
    Receives: Binary audio data (int16, 16kHz)
    Produces: Transcription results forwarded to agent (placeholder)
    """
    assert stt_manager is not None, "STT manager not initialized"
    manager = stt_manager  # Local reference for type checker
    
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"STT session {session_id} connected")
    
    # Register session and get result queue
    result_queue = manager.register(session_id)
    
    async def receive_audio():
        """Receive audio from WebSocket and stream to STT worker."""
        try:
            while True:
                audio_bytes = await websocket.receive_bytes()
                manager.stream_audio(session_id, audio_bytes)
        except WebSocketDisconnect:
            logger.info(f"STT session {session_id} disconnected")
        except Exception as e:
            logger.error(f"Audio receive error: {e}")
    
    async def process_results():
        """Process transcription results and forward to agent."""
        try:
            while True:
                result = await result_queue.get()
                
                if result.get("type") == "result":
                    transcription = result.get("text", "")
                    lang = result.get("lang")
                    logger.info(f"[{session_id}] Transcription: {transcription} (lang={lang})")
                    
                    # TODO: Forward to LangChain agent
                    # await agent.process(session_id, transcription, lang)
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Result processing error: {e}")
    
    # Run both tasks concurrently
    receive_task = asyncio.create_task(receive_audio())
    process_task = asyncio.create_task(process_results())
    
    try:
        # Wait for receive task to complete (disconnect)
        await receive_task
    finally:
        # Cleanup
        process_task.cancel()
        manager.unregister(session_id)
        logger.info(f"STT session {session_id} cleaned up")