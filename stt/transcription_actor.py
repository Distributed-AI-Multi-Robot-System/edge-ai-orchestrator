import ray
import numpy as np
from faster_whisper import WhisperModel


@ray.remote(num_cpus=2)
class TranscriptionActor:
    """
    Ray actor that hosts a Whisper model for transcription.
    
    Multiple instances can be pooled (e.g., 2x base for accuracy, 2x tiny for latency).
    Each actor owns its model instance to avoid contention.
    """
    
    def __init__(self, model_name: str = "base", device: str = "cpu"):
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[TranscriptionActor] Loading model '{model_name}' on {device}...")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=2)
        self.model_name = model_name
        print(f"[TranscriptionActor] Model '{model_name}' ready")

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
            
            # Collect text from segments
            text_parts = [s.text.strip() for s in segments if s.text.strip()]
            result_text = " ".join(text_parts)
            
            return (result_text, info.language)
            
        except Exception as e:
            print(f"[TranscriptionActor] Error transcribing: {e}")
            return ("", None)
