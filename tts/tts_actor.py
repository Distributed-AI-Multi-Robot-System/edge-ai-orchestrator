import ray
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator
from nltk import sent_tokenize, download

VALID_ENDINGS = {'.', '!', '?', ';', ':'}


@ray.remote
class TTSActor:
    def __init__(self):
        self.text_buffer = ""
        try:
            download('punkt_tab')
        except Exception as e:
            print(f"[TTSActor] NLTK download warning: {e}")
            pass

        print("[TTSActor] Initialized and ready")

    async def synthesize_text(self, text: str, finalize: bool, tts_deployment_handle: DeploymentHandle):
        """
        Synthesize text to speech using the PiperDeployment.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            Async generator yielding audio chunks as bytes
        """
        # 1. Update buffer
        self.text_buffer += text

        # 2. Handle finalization (flush everything)
        if finalize:
            if not self.text_buffer.strip():
                return
            
            # Synthesize EVERYTHING left in the buffer
            text_to_process = self.text_buffer
            self.text_buffer = "" # Clear immediately
            
            tts_stream_handle = tts_deployment_handle.options(stream=True)
            gen: DeploymentResponseGenerator = tts_stream_handle.synthesize.remote(text_to_process)

            async for audio_chunk in gen:
                yield audio_chunk
            return

        # 3. Normal Processing (Tokenize and find complete sentences)
        sentences = sent_tokenize(self.text_buffer)
        if not sentences:
            return
        
        last_sentence = sentences[-1].strip()
        if not last_sentence or last_sentence[-1] not in VALID_ENDINGS:
            # It's incomplete! Remove it from the list and store it.
            incomplete_segment = sentences.pop()

            self.text_buffer = incomplete_segment
        else:
            self.text_buffer = ""

        text_to_synthesize = " ".join(sentences)
        if not text_to_synthesize:
            return
        
        tts_stream_handle = tts_deployment_handle.options(stream=True)
        gen: DeploymentResponseGenerator = tts_stream_handle.synthesize.remote(text_to_synthesize)

        async for audio_chunk in gen:
            yield audio_chunk   
