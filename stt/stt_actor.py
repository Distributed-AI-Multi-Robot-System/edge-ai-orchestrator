import asyncio
import ray
from ray import serve
import numpy as np
import logging
from silero_vad import load_silero_vad, VADIterator

# --- Configuration ---
SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
MIN_PIPELINE_DURATION = 3.0  # Seconds before offloading to background model

logger = logging.getLogger(__name__)


@ray.remote
class STTActor:
    """
    Ray actor that manages VAD and transcription for a single session.
    """
    
    def __init__(self, whsper_base_deployment_name: str, whsper_tiny_deployment_name: str):
        try:
            self.base_whisper = serve.get_app_handle(whsper_base_deployment_name)
            self.tail_whisper = serve.get_app_handle(whsper_tiny_deployment_name)
        except Exception as e:
            print(f"Error connecting to Serve: {e}")
            # It's often good practice to retry or fail gracefully here
            raise e
        self.pending_futures = []
        
        # Load VAD model
        self.vad_model = load_silero_vad(onnx=True)
        
        # VAD iterators
        self.vad_controller = VADIterator(
            self.vad_model,
            threshold=0.6,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=700,
            speech_pad_ms=30
        )
        self.vad_pause = VADIterator(
            self.vad_model,
            threshold=0.3,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=200,
            speech_pad_ms=0
        )

        self.remainder = np.array([], dtype=np.float32)
        
        self._reset_state()
        self.lang_hint = None
        print("[STTActor] Initialized and ready")

    def _reset_state(self):
        """Reset all session state for a new sentence."""
        self.sentence_buffer: list[np.ndarray] = []
        self.pending_count: int = 0
        self.transcription_parts: list[str] = []
        self.is_recording = False
        self.lang_hint: str | None = None
        self.vad_pause.reset_states()
        self.pending_futures = []
        # [NOTE] We do NOT reset self.remainder here. It must survive across sentences.


    async def compute_audio(self, chunk_bytes: bytes) -> tuple[str, str] | None:
        """
        Process incoming audio chunk with VAD and manage transcription.
        """
        # Convert int16 bytes to float32
        audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        new_audio = audio_int16.astype(np.float32) / 32768.0

        # Prepend remainder from previous call to fix fragmentation
        if self.remainder.size > 0:
            audio_float32 = np.concatenate([self.remainder, new_audio])
        else:
            audio_float32 = new_audio

        # Calculate valid chunks and save the new remainder
        n_samples = len(audio_float32)
        n_chunks = n_samples // VAD_WINDOW_SIZE
        remainder_start = n_chunks * VAD_WINDOW_SIZE

        # Save the "leftover" bytes for the next call
        self.remainder = audio_float32[remainder_start:]
        
        # Loop only up to remainder_start (guarantees full 512 chunks)
        for i in range(0, remainder_start, VAD_WINDOW_SIZE):
            chunk = audio_float32[i:i + VAD_WINDOW_SIZE]
            
            # This check is now theoretically redundant but good for safety
            if len(chunk) < VAD_WINDOW_SIZE:
                break
            
            controller_event = self.vad_controller(chunk, return_seconds=True)
            pause_event = self.vad_pause(chunk, return_seconds=True)
            
            # --- SPEECH START ---
            if controller_event and 'start' in controller_event:
                if self.is_recording:
                    self._reset_state()
                self.is_recording = True
            
            # Buffer audio while recording
            if self.is_recording:
                self.sentence_buffer.append(chunk)
            
            # --- BACKGROUND OFFLOAD (on pause) ---
            current_duration = (len(self.sentence_buffer) * VAD_WINDOW_SIZE) / SAMPLE_RATE
            
            if self.is_recording and pause_event and 'end' in pause_event:
                if current_duration > MIN_PIPELINE_DURATION and self.sentence_buffer:
                    audio_segment = np.concatenate(self.sentence_buffer)
                    self.sentence_buffer = []
                    
                    future = self.base_whisper.transcribe.remote(audio_segment, self.lang_hint)
                    self.pending_futures.append(future)
                
                self.vad_pause.reset_states()
            
            # --- SPEECH END ---
            if controller_event and 'end' in controller_event:
                self.is_recording = False
                return await self._finalize_transcription()
        
        return None

    async def _finalize_transcription(self) -> tuple[str, str] | None:
        """
        Finalize transcription: process tail, collect all results, reset state.
        """
        
        # [CHANGE 5] Peek at background tasks to get language hint BEFORE sending tail
        if self.pending_futures:
            try:
                # Wait briefly (50ms) to see if background model has a result
                done, _ = await asyncio.wait(self.pending_futures, timeout=0.05)
                for task in done:
                    # If any task finished, grab the language
                    _, lang = await task 
                    if lang and not self.lang_hint:
                        self.lang_hint = lang
            except Exception:
                pass

        if self.sentence_buffer:
            audio_tail = np.concatenate(self.sentence_buffer)
            future = self.tail_whisper.transcribe.remote(audio_tail, self.lang_hint)
            self.pending_futures.append(future)
        
        # Collect all pending results
        all_parts = []
        detected_lang = self.lang_hint
        
        try:
            if self.pending_futures:
                results = await asyncio.gather(*self.pending_futures)

                for text, lang in results:
                    if text:
                        all_parts.append(text)
                    if lang and not detected_lang:
                        detected_lang = lang
                        
        except Exception as e:
            logger.error(f"[STTActor] Error collecting results: {e}")
        
        # Build final transcription
        full_transcription = " ".join(all_parts).strip()
        
        # Reset state for next sentence
        self._reset_state()
        
        return (full_transcription, detected_lang or "unknown") if full_transcription else None