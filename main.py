import uuid
import logging
import ray
from ray import serve
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from agent.agent_manager import AgentManager

from stt import STTManager
from tts.tts_manager import TTSManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global manager instances
stt_manager: STTManager | None = None
tts_manager: TTSManager | None = None
agent_manager: AgentManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stt_manager
    global tts_manager
    global agent_manager
    
    # Startup: Initialize Ray and STT manager
    logger.info("Initializing Ray...")
    ray.init(ignore_reinit_error=True)
    serve.start(http_options={"port": 8001})
    
    logger.info("Starting STT manager...")
    stt_manager = STTManager()
    stt_manager.start()

    logger.info("Starting TTS manager...")
    tts_manager = TTSManager()
    tts_manager.start()

    logger.info("Starting Agent manager...")
    agent_manager = AgentManager()
    
    yield
    logger.info("Shutting down Ray...")
    ray.shutdown()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/stt")
async def stt_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Speech-to-Text.
    
    Receives: Binary audio data (int16, 16kHz)
    Produces: Transcription results forwarded to agent (placeholder)
    """
    assert stt_manager is not None, "STT manager not initialized"
    assert tts_manager is not None, "TTS manager not initialized"
    manager = stt_manager  # Local reference for type checker
    
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"STT session {session_id} connected")
    
    # Register session (spawns STTActor)
    stt_actor = manager.register(session_id)
    tts_actor = tts_manager.register(session_id=session_id)
    
    try:
        while True:
            # Receive audio chunk from client
            audio_bytes = await websocket.receive_bytes()
            
            # Stream to STTActor and check for transcription result
            result = await manager.stream_audio(stt_actor, audio_bytes)
            
            if not result:
                continue  # No transcription yet

            transcription, lang = result
            
            # Send result back to client
            await websocket.send_json({
                "type": "result",
                "text": transcription,
                "lang": lang
            })

            # language mapping for TTS
            language_map = {
                "de": "german",
                "en": "english",
                "fr": "french",
                "it": "italian"
            }
            tts_actor_lang = language_map.get(lang, "english")

            
            tts_handle = tts_manager.get_deployment_handle(language=lang)
            full_agent_response = []

            async for agent_text_chunk in agent_manager.stream_agent_response(transcription, session_id):
                # Stream each chunk to TTSActor
                tts_stream = tts_actor.synthesize_text.remote(
                    text=agent_text_chunk,
                    tts_deployment_handle=tts_handle,
                    finalize=False,
                    language=tts_actor_lang
                )

                full_agent_response.append(agent_text_chunk)
                
                async for chunk_ref in tts_stream:
                    # DEBUG: Check exactly what Ray is streaming back
                    chunk = await chunk_ref
                    
                    # Only send if it is valid bytes
                    await websocket.send_bytes(chunk)

            await websocket.send_json({
                "type": "result",
                "text": "".join(full_agent_response),
                "lang": lang
            })
                
    except WebSocketDisconnect:
        logger.info(f"STT session {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup session
        logger.info(f"STT session {session_id} cleaned up")