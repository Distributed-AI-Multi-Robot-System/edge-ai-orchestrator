import ray
from faster_whisper import WhisperModel
import numpy as np

## fixed numer of transcription actors to handle audio transcription tasks from multiple stt_actor (one stt worker per session)
@ray.remote
class TailTranscriptionActor:
    def __init__(self):
        self.model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads= 2)


    def transcribe(self, audio_buffer, language=None):
        """
        Returns tuple: (list_of_text_segments, detected_language_code)
        """
        if not audio_buffer: return ([], None)
        full_audio = np.concatenate(audio_buffer)
        
        segments, info = self.model.transcribe(
            audio=full_audio, 
            beam_size=1, 
            language=language, 
            condition_on_previous_text=False
        )
        result_text = [s.text.strip() for s in segments if len(s.text.strip()) > 1]
        return result_text, info.language