import multiprocessing
import traceback
import numpy as np
import concurrent.futures
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator
from stt.device_type import DeviceType

# --- CONFIG ---
SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
MIN_PIPELINE_DURATION = 3.0  # Seconds before offloading to background model

class SessionState:
    """Storing the state for a User/Session"""
    def __init__(self, vad_model):
        self.vad_controller = VADIterator(vad_model, threshold=0.6, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=1000, speech_pad_ms=30)
        self.vad_worker_pause = VADIterator(vad_model, threshold=0.3, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=300, speech_pad_ms=0)
        
        self.sentence_buffer = [] 
        self.pending_futures = []
        self.is_recording = False
        self.lang_hint = None

def worker_logic(
    input_queue, 
    output_queue, 
    device_type: DeviceType = DeviceType.CUDA,
    model_bg_name: str = "base",
    model_tail_name: str = "tiny"
):
    device = device_type.value
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[Worker] Initializing on {device}...")
    
    # 1. Load models (GPU memory allocated here)
    # Background model (accuracy)
    print(f"[Worker] Loading Background Model ({model_bg_name})...")
    model_bg = WhisperModel(model_bg_name, device=device, compute_type=compute_type)
    
    # Tail model (latency)
    print(f"[Worker] Loading Tail Model ({model_tail_name})...")
    model_tail = WhisperModel(model_tail_name, device=device, compute_type=compute_type)
    
    # Load VAD (CPU, lightweight)
    vad_model = load_silero_vad(onnx=True)
    
    # Session state management (session_id -> SessionState)
    sessions = {}
    
    # Executor for parallel inference within the process
    # IMPORTANT: Prevents VAD from blocking while Whisper is processing
    # Whisper releases the GIL, so threads work well here
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def transcribe_segment(model, audio, lang):
        """Helper function for ThreadPool"""
        segments, info = model.transcribe(audio, beam_size=5, language=lang, condition_on_previous_text=False)
        text = " ".join([s.text.strip() for s in segments if s.text.strip()])
        return text, info.language

    print("[Worker] Ready & Waiting for Stream...")

    while True:
        try:
            # 1. Get message from queue
            msg = input_queue.get()
            
            if msg is None: break # Poison Pill
            
            msg_type = msg.get("type")
            session_id = msg.get("session_id")

            # Session Management
            if session_id not in sessions:
                sessions[session_id] = SessionState(vad_model)
            
            session = sessions[session_id]

            # --- CASE A: Process Audio Chunk ---
            if msg_type == "audio":
                audio_bytes = msg.get("data")
                
                # Convert int16 bytes to float32 (normalization for Whisper)
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_chunk = audio_int16.astype(np.float32) / 32768.0
                
                # VAD Loop (iterate over chunk)
                for i in range(0, len(audio_chunk), VAD_WINDOW_SIZE):
                    chunk = audio_chunk[i: i+VAD_WINDOW_SIZE]
                    if len(chunk) < VAD_WINDOW_SIZE: break

                    controller_event = session.vad_controller(chunk, return_seconds=True)
                    worker_pause_event = session.vad_worker_pause(chunk, return_seconds=True)

                    # START
                    if controller_event and 'start' in controller_event:
                        session.is_recording = True
                        session.sentence_buffer = []
                        session.pending_futures = []
                        session.lang_hint = None
                        session.vad_worker_pause.reset_states()
                    
                    if session.is_recording:
                        session.sentence_buffer.append(chunk)

                    # BACKGROUND OFFLOAD LOGIC
                    current_duration = (len(session.sentence_buffer) * VAD_WINDOW_SIZE) / SAMPLE_RATE
                    
                    if session.is_recording and (worker_pause_event and 'end' in worker_pause_event):
                        if current_duration > MIN_PIPELINE_DURATION:
                            # Copy buffer and send to ThreadPool
                            audio_to_process = np.concatenate(session.sentence_buffer)
                            session.sentence_buffer = []  # Flush buffer
                            
                            # Submit to Background Model
                            future = executor.submit(transcribe_segment, model_bg, audio_to_process, session.lang_hint)
                            session.pending_futures.append(future)
                            
                        session.vad_worker_pause.reset_states()

                    # END (Sentence Final)
                    if controller_event and 'end' in controller_event:
                        session.is_recording = False
                        
                        # Process remaining audio (tail) with FAST model
                        if session.sentence_buffer:
                            audio_tail = np.concatenate(session.sentence_buffer)
                            # Submit to Tail Model
                            future = executor.submit(transcribe_segment, model_tail, audio_tail, session.lang_hint)
                            session.pending_futures.append(future)
                        
                        # --- SYNCHRONIZATION (Wait for completion) ---
                        # Wait until all parts of the sentence are processed.
                        # Short blocking is acceptable here since inference is already running,
                        # we're just collecting results.
                        
                        full_text_parts = []
                        final_lang = session.lang_hint
                        
                        for f in session.pending_futures:
                            text, lang = f.result()  # Wait for result
                            full_text_parts.append(text)
                            if not final_lang and lang:
                                final_lang = lang
                                session.lang_hint = lang  # Update hint for next chunks
                        
                        full_transcription = " ".join(full_text_parts).strip()
                        
                        if full_transcription:
                            # Send final result to output queue
                            output_queue.put({
                                "session_id": session_id,
                                "type": "result",
                                "text": full_transcription,
                                "lang": final_lang
                            })
                        
                        # Reset
                        session.sentence_buffer = []
                        session.pending_futures = []
                        session.vad_worker_pause.reset_states()

            # --- CASE B: Session Cleanup ---
            elif msg_type == "disconnect":
                if session_id in sessions:
                    del sessions[session_id]

        except Exception as e:
            print(f"[Worker Error] {e}")
            traceback.print_exc()