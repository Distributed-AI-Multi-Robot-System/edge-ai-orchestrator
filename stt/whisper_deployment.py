from ray import serve
import numpy as np
from faster_whisper import WhisperModel
import logging

logger = logging.getLogger("ray.serve")

@serve.deployment(autoscaling_config={"min_replicas": 1, "max_replicas": 4}, ray_actor_options={"num_cpus": 2})
class WhisperDeployment:
    def __init__(self, model_name: str, device: str = "cpu", cpu_threads: int = 2):
        self.device = device
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(f"Loading {model_name} model on {device}...")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
    def transcribe(self, audio_float32: np.ndarray, language: str | None = None) -> tuple[str, str | None]:
        """
        Transcribe audio segment.
        
        Args:
            audio_float32: Audio samples as float32 numpy array (normalized -1 to 1)
            language: Optional language hint (e.g., "en", "de")
            
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        if audio_float32 is None or len(audio_float32) == 0:
            return ("", None)
        
        try:
            segments, info = self.model.transcribe(
                audio=audio_float32,
                beam_size=1,
                language=language,
                condition_on_previous_text=False
            )
            text = " ".join([s.text.strip() for s in segments if s.text.strip()])
            return (text, info.language)
        except Exception as e:
            logger.error(f"Error transcribing: {e}")
            return ("", None)