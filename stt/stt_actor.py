import ray
from silero_vad import load_silero_vad, VADIterator
import asyncio
import numpy as np

SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
MIN_PIPELINE_DURATION = 3.0  # Seconds before offloading to background model

@ray.remote
class STTActor:
    def __init__(self):
        self.vad_model = load_silero_vad(onnx=True)

        self.vad_controller = VADIterator(
            self.vad_model, threshold=0.6, sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=1000, 
            speech_pad_ms=30
        )
        self.vad_worker_pause = VADIterator(
            self.vad_model, threshold=0.3, sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=300, 
            speech_pad_ms=0 
        )

        self.model_lang = None
        self.sentence_buffer = []
        self.pending_tasks = [] 
        self.is_recording = False 
        
        self.loop = asyncio.get_running_loop()

    async def transcribe(self, model_instance, language=None):
        """
        Returns tuple: (list_of_text_segments, detected_language_code)
        """
        if not self.sentence_buffer: return ([], None)
        full_audio = np.concatenate(self.sentence_buffer)
        
        def _run():
            segments, info = model_instance.transcribe(
                audio=full_audio, 
                beam_size=1, 
                language=language, 
                condition_on_previous_text=False
            )
            result_text = [s.text.strip() for s in segments if len(s.text.strip()) > 1]
            return result_text, info.language

        return await self.loop.run_in_executor(None, _run)