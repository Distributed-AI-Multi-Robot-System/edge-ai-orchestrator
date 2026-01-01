import ray
import logging
import os

from stt.stt_actor import STTActor
from stt.transcription_router import TranscriptionRouter

logger = logging.getLogger(__name__)


class STTManager:
    """
    Manages Ray-based STT infrastructure.
    
    - Creates and manages TranscriptionActor pools (base + tiny models)
    - Spawns STTActor per WebSocket session
    - Provides async interface for audio streaming
    """
    
    def __init__(self):
        # Load configuration from environment
        self.device = os.getenv("STT_DEVICE", "cpu").lower()
        self.model_bg = os.getenv("STT_MODEL_BG", "base")
        self.model_tail = os.getenv("STT_MODEL_TAIL", "tiny")
        self.pool_base_size = int(os.getenv("STT_POOL_BASE", "2"))
        self.pool_tiny_size = int(os.getenv("STT_POOL_TINY", "2"))
        self.router = None
        
        logger.info(
            f"STTManager config: device={self.device}, "
            f"bg_model={self.model_bg} (pool={self.pool_base_size}), "
            f"tail_model={self.model_tail} (pool={self.pool_tiny_size})"
        )

    def start(self):
        logger.info("Starting Transcription Router...")
        # Start the centralized router
        self.router = TranscriptionRouter.remote(
            base_size=self.pool_base_size,
            tiny_size=self.pool_tiny_size,
            device=self.device
        )
        logger.info("STT Infrastructure ready.")


    def register(self, session_id: str):
        """
        Register a new session and spawn its STTActor.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            STTActor handle for the session
        """
        # Spawn new STTActor with pool references
        actor = STTActor.options(name=session_id, get_if_exists=True).remote(self.router)
        logger.debug(f"Session {session_id} registered")
        
        return actor


    async def stream_audio(self, actor, chunk_bytes: bytes) -> str | None:
        """
        Stream audio chunk to session's STTActor.
        
        Args:
            session_id: Session identifier
            chunk_bytes: Raw audio bytes (int16, 16kHz)
            
        Returns:
            Transcription string when speech ends, None otherwise
        """
    
        try:
            result = await actor.compute_audio.remote(chunk_bytes)
            return result
        except Exception as e:
            logger.error(f"Error streaming audio to STTActor: {e}")
            return None