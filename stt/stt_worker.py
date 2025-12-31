import multiprocessing
import queue
import traceback
import numpy as np
import concurrent.futures
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator
import device_type as dt

# --- CONFIG ---
SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512
MIN_PIPELINE_DURATION = 3.0 # Sekunden bevor wir "offloaden"

class SessionState:
    """Storing the state for a User/Session"""
    def __init__(self, vad_model):
        self.vad_controller = VADIterator(vad_model, threshold=0.6, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=1000, speech_pad_ms=30)
        self.vad_worker_pause = VADIterator(vad_model, threshold=0.3, sampling_rate=SAMPLE_RATE, min_silence_duration_ms=300, speech_pad_ms=0)
        
        self.sentence_buffer = [] 
        self.pending_futures = []
        self.is_recording = False
        self.lang_hint = None

def worker_logic(input_queue, output_queue, device_type=(dt.DeviceType.CUDA)):
    device = device_type.value
    print(f"[Worker] Initialisiere auf {device}...")
    
    # 1. Modelle laden (GPU Speicher wird hier belegt)
    # Hintergrund-Modell (Genauigkeit)
    print("[Worker] Lade Background Model (medium)...")
    model_bg = WhisperModel("medium", device=device, compute_type="float16")
    
    # Tail-Modell (Latenz)
    print("[Worker] Lade Tail Model (tiny)...")
    model_tail = WhisperModel("tiny", device=device, compute_type="float16")
    
    # VAD laden (CPU, sehr leicht)
    vad_model = load_silero_vad(onnx=True)
    
    # State-Management für Sessions (session_id -> SessionState)
    sessions = {}
    
    # Executor für parallele Inferenz innerhalb des Prozesses
    # WICHTIG: Damit VAD nicht blockiert, wenn Whisper rechnet.
    # Whisper gibt den GIL frei, daher funktionieren Threads hier gut.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def transcribe_segment(model, audio, lang):
        """Hilfsfunktion für den ThreadPool"""
        segments, info = model.transcribe(audio, beam_size=5, language=lang, condition_on_previous_text=False)
        text = " ".join([s.text.strip() for s in segments if s.text.strip()])
        return text, info.language

    print("[Worker] Ready & Waiting for Stream...")

    while True:
        try:
            # 1. Daten holen
            msg = input_queue.get()
            
            if msg is None: break # Poison Pill
            
            msg_type = msg.get("type")
            session_id = msg.get("session_id")

            # Session Management
            if session_id not in sessions:
                sessions[session_id] = SessionState(vad_model)
            
            session = sessions[session_id]

            # --- CASE A: Audio Chunk verarbeiten ---
            if msg_type == "audio":
                audio_chunk = msg.get("data")
                
                # VAD Loop (über den Chunk iterieren)
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
                            # Wir kopieren den Buffer und schicken ihn an den ThreadPool
                            audio_to_process = np.concatenate(session.sentence_buffer)
                            session.sentence_buffer = [] # Flush buffer
                            
                            # Submit an Background Model (Medium)
                            future = executor.submit(transcribe_segment, model_bg, audio_to_process, session.lang_hint)
                            session.pending_futures.append(future)
                            
                        session.vad_worker_pause.reset_states()

                    # END (Sentence Final)
                    if controller_event and 'end' in controller_event:
                        session.is_recording = False
                        
                        # Den Rest (Tail) mit dem FAST Model verarbeiten
                        if session.sentence_buffer:
                            audio_tail = np.concatenate(session.sentence_buffer)
                            # Submit an Tail Model (Tiny)
                            future = executor.submit(transcribe_segment, model_tail, audio_tail, session.lang_hint)
                            session.pending_futures.append(future)
                        
                        # --- SYNCHRONISATION (Wait for completion) ---
                        # Hier warten wir, bis alle Teile des Satzes fertig sind.
                        # Da wir im Worker sind, ist kurzes Blockieren hier okay, 
                        # oder wir könnten das Ergebnis asynchron pushen. 
                        # Für Einfachheit blockieren wir kurz den Worker-Loop für diesen User,
                        # aber da Inferenz schon läuft, ist das nur "einsammeln".
                        
                        full_text_parts = []
                        final_lang = session.lang_hint
                        
                        for f in session.pending_futures:
                            text, lang = f.result() # Wartet auf Ergebnis
                            full_text_parts.append(text)
                            if not final_lang and lang:
                                final_lang = lang
                                session.lang_hint = lang # Update hint for next chunks
                        
                        full_transcription = " ".join(full_text_parts).strip()
                        
                        if full_transcription:
                            # FINALES ERGEBNIS AN API SENDEN
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