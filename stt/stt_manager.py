import logging
import os
from ray import serve

from stt.stt_actor import STTActor
from stt.whisper_deployment import WhisperDeployment
logger = logging.getLogger(__name__)

BASE_APP_NAME = "base_whisper_deployment"
TAIL_APP_NAME = "tail_whisper_deployment"


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
        logger.info("Starting Ray Serve Deployments...")
        base_app = WhisperDeployment.options(
            name="WhisperBase", 
            ray_actor_options={"num_cpus": 3}
        ).bind(model_name="base", device=self.device, cpu_threads=3)
        
        tail_app = WhisperDeployment.options(
            name="WhisperTail",
            ray_actor_options={"num_cpus": 2}
        ).bind(model_name="tiny", device=self.device, cpu_threads=2)

        serve.run(base_app, name=BASE_APP_NAME, route_prefix="/base")
        serve.run(tail_app, name=TAIL_APP_NAME, route_prefix="/tail")
        
        logger.info("Ray Serve Deployments Ready: 'WhisperBase' and 'WhisperTiny'")


    def register(self, session_id: str):
        """
        Register a new session and spawn its STTActor.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            STTActor handle for the session
        """
        # Spawn new STTActor with pool references
        actor = STTActor.options(name=session_id, get_if_exists=True).remote(
            BASE_APP_NAME, 
            TAIL_APP_NAME
        )
        logger.debug(f"Session {session_id} registered")
        
        return actor


    async def stream_audio(self, actor, chunk_bytes: bytes) -> tuple[str, str] | None:
        """
        Stream audio chunk to session's STTActor.
        
        Args:
            session_id: Session identifier
            chunk_bytes: Raw audio bytes (int16, 16kHz)
            
        Returns:
            Transcription string when speech ends, None otherwise
        """
    
        try:
            return await actor.compute_audio.remote(chunk_bytes)
        except Exception as e:
            logger.error(f"Error streaming audio to STTActor: {e}")
            return None