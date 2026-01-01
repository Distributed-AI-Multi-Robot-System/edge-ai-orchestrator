import ray
import logging
import os
from ray.util.actor_pool import ActorPool

from stt.transcription_actor import TranscriptionActor
from stt.stt_actor import STTActor

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
        
        # Actor pools (initialized in start())
        self.base_pool: ActorPool | None = None
        self.tiny_pool: ActorPool | None = None
        
        logger.info(
            f"STTManager config: device={self.device}, "
            f"bg_model={self.model_bg} (pool={self.pool_base_size}), "
            f"tail_model={self.model_tail} (pool={self.pool_tiny_size})"
        )

    def start(self):
        """Initialize transcription actor pools."""
        logger.info("Creating transcription actor pools...")
        
        # Create base model pool (for accuracy, background transcription)
        base_actors = [
            TranscriptionActor.remote(self.model_bg, self.device)
            for _ in range(self.pool_base_size)
        ]
        self.base_pool = ActorPool(base_actors)
        logger.info(f"Base pool created: {self.pool_base_size}x {self.model_bg}")
        
        # Create tiny model pool (for latency, tail transcription)
        tiny_actors = [
            TranscriptionActor.remote(self.model_tail, self.device)
            for _ in range(self.pool_tiny_size)
        ]
        self.tiny_pool = ActorPool(tiny_actors)
        logger.info(f"Tiny pool created: {self.pool_tiny_size}x {self.model_tail}")
        
        logger.info("STT manager started")


    def register(self, session_id: str):
        """
        Register a new session and spawn its STTActor.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            STTActor handle for the session
        """
        # Spawn new STTActor with pool references
        actor = STTActor.options(name=session_id, get_if_exists=True).remote(self.base_pool, self.tiny_pool)
        logger.debug(f"Session {session_id} registered")
        
        return actor

    def unregister(self, session_id: str):
        """
        Unregister a session and kill its actor.
        
        Args:
            session_id: Session to unregister
        """
        actor = ray.get_actor(session_id)
        if not actor:
            logger.warning(f"No actor found for session {session_id} during unregister")
            return
        
        ray.kill(actor)

    async def stream_audio(self, session_id: str, chunk_bytes: bytes) -> str | None:
        """
        Stream audio chunk to session's STTActor.
        
        Args:
            session_id: Session identifier
            chunk_bytes: Raw audio bytes (int16, 16kHz)
            
        Returns:
            Transcription string when speech ends, None otherwise
        """
        actor = ray.get_actor(session_id)
        if not actor:
            logger.error(f"Session {session_id} not registered")
            return None
        
        try:
            # Call actor and await result
            result = await actor.compute_audio.remote(chunk_bytes)
            return result
        except Exception as e:
            logger.error(f"Error processing audio for session {session_id}: {e}")
            return None