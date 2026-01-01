import ray
import numpy as np
import logging
from silero_vad import load_silero_vad, VADIterator
from ray.util.actor_pool import ActorPool

# --- Configuration ---
SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
MIN_PIPELINE_DURATION = 3.0  # Seconds before offloading to background model

logger = logging.getLogger(__name__)


@ray.remote
class STTActor:
    """
    Ray actor that manages VAD and transcription for a single session.
    
    Each WebSocket session gets its own STTActor instance.
    The actor processes audio chunks, detects speech boundaries via VAD,
    and coordinates with shared TranscriptionActor pools for Whisper inference.
    
    compute_audio() returns:
        - None: Speech ongoing, no complete sentence yet
        - str: Full transcription when speech ends
    """
    
    def __init__(self, base_pool: ActorPool, tiny_pool: ActorPool):
        """
        Initialize STT actor with transcription pools.
        
        Args:
            base_pool: ActorPool of TranscriptionActors with "base" model (accuracy)
            tiny_pool: ActorPool of TranscriptionActors with "tiny" model (latency)
        """
        self.base_pool = base_pool
        self.tiny_pool = tiny_pool
        
        # Load VAD model (lightweight, ~2MB, CPU-only)
        self.vad_model = load_silero_vad(onnx=True)
        
        # VAD iterators for speech detection
        self.vad_controller = VADIterator(
            self.vad_model,
            threshold=0.6,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=1000,
            speech_pad_ms=30
        )
        self.vad_pause = VADIterator(
            self.vad_model,
            threshold=0.3,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=300,
            speech_pad_ms=0
        )
        
        # Session state
        self._reset_state()
        print("[STTActor] Initialized and ready")

    def _reset_state(self):
        """Reset all session state for a new sentence."""
        self.sentence_buffer: list[np.ndarray] = []
        self.pending_count: int = 0  # Count of pending pool submissions
        self.transcription_parts: list[str] = []
        self.is_recording = False
        self.lang_hint: str | None = None
        self.vad_pause.reset_states()

    def compute_audio(self, chunk_bytes: bytes) -> str | None:
        """
        Process incoming audio chunk with VAD and manage transcription.
        
        Args:
            chunk_bytes: Raw audio bytes (int16, 16kHz)
            
        Returns:
            None if speech is ongoing, full transcription string when speech ends
        """
        # Convert int16 bytes to float32
        audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        result = None
        
        # Process audio in VAD_WINDOW_SIZE chunks
        for i in range(0, len(audio_float32), VAD_WINDOW_SIZE):
            chunk = audio_float32[i:i + VAD_WINDOW_SIZE]
            if len(chunk) < VAD_WINDOW_SIZE:
                break
            
            # Run VAD
            controller_event = self.vad_controller(chunk, return_seconds=True)
            pause_event = self.vad_pause(chunk, return_seconds=True)
            
            # --- SPEECH START ---
            if controller_event and 'start' in controller_event:
                self.is_recording = True
                self._reset_state()
                self.is_recording = True  # Re-set after reset
            
            # Buffer audio while recording
            if self.is_recording:
                self.sentence_buffer.append(chunk)
            
            # --- BACKGROUND OFFLOAD (on pause) ---
            current_duration = (len(self.sentence_buffer) * VAD_WINDOW_SIZE) / SAMPLE_RATE
            
            if self.is_recording and pause_event and 'end' in pause_event:
                if current_duration > MIN_PIPELINE_DURATION:
                    # Offload accumulated audio to background model
                    audio_segment = np.concatenate(self.sentence_buffer)
                    self.sentence_buffer = []
                    
                    # Submit to base pool (fire-and-forget, collect later)
                    self.base_pool.submit(
                        lambda actor, args: actor.transcribe.remote(*args),
                        (audio_segment, self.lang_hint)
                    )
                    # Track pending result
                    self.pending_count += 1
                
                self.vad_pause.reset_states()
            
            # --- SPEECH END ---
            if controller_event and 'end' in controller_event:
                self.is_recording = False
                result = self._finalize_transcription()
        
        return result

    def _finalize_transcription(self) -> str:
        """
        Finalize transcription: process tail, collect all results, reset state.
        
        Returns:
            Complete transcription string
        """
        # Submit remaining audio (tail) to tiny model for low latency
        tail_submitted = False
        if self.sentence_buffer:
            audio_tail = np.concatenate(self.sentence_buffer)
            self.tiny_pool.submit(
                lambda actor, args: actor.transcribe.remote(*args),
                (audio_tail, self.lang_hint)
            )
            tail_submitted = True
        
        # Collect all pending results from pools
        all_parts = []
        detected_lang = self.lang_hint
        
        try:
            # Get results from base pool (background transcriptions)
            for _ in range(self.pending_count):
                if self.base_pool.has_next():
                    text, lang = self.base_pool.get_next_unordered()
                    if text:
                        all_parts.append(text)
                    if lang and not detected_lang:
                        detected_lang = lang
                        self.lang_hint = lang
            
            # Get tail result from tiny pool
            if tail_submitted and self.tiny_pool.has_next():
                text, lang = self.tiny_pool.get_next_unordered()
                if text:
                    all_parts.append(text)
                if lang and not detected_lang:
                    detected_lang = lang
                    
        except Exception as e:
            print(f"[STTActor] Error collecting results: {e}")
        
        # Build final transcription
        full_transcription = " ".join(all_parts).strip()
        
        # Reset state for next sentence
        self._reset_state()
        
        return full_transcription if full_transcription else ""
