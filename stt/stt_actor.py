import ray
from silero_vad import load_silero_vad, VADIterator
import asyncio
import numpy as np
from ray.util.actor_pool import ActorPool

SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
MIN_PIPELINE_DURATION = 3.0  # Seconds before offloading to background model

@ray.remote
class STTActor:
    def __init__(self, transcription_actor_pool: ActorPool):
        self.transcription_actor_pool: ActorPool = transcription_actor_pool
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


    async def compute_audio(self, audio_chunk: np.ndarray):
        """Process incoming audio chunk with VAD and manage sentence buffer."""
