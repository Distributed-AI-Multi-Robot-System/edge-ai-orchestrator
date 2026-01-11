import ray
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator
from nltk import sent_tokenize, download

download('punkt_tab')
valid_endings = {'.', '!', '?', ';', ':'}


@ray.remote
class TTSActor:
    def __init__(self):
        self.text_buffer = ""
        print("[TTSActor] Initialized and ready")

    async def synthesize_text(self, text: str, finalize: bool, tts_deployment_handle: DeploymentHandle):
        """
        Synthesize text to speech using the PiperDeployment.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            Async generator yielding audio chunks as bytes
        """
        self.text_buffer += text

        if finalize:
            tts_stream_handle = tts_deployment_handle.options(stream=True)
            gen: DeploymentResponseGenerator = tts_stream_handle.synthesize.remote(text)

            async for audio_chunk in gen:
                yield audio_chunk

            return

        
        sentences = sent_tokenize(self.text_buffer)
        if not sentences:
            return
        
        last_sentence = sentences[-1].strip()
        if not last_sentence or last_sentence[-1] not in valid_endings:
            # It's incomplete! Remove it from the list and store it.
            incomplete_buffer = sentences.pop()

            self.text_buffer = incomplete_buffer
        else:
            self.text_buffer = ""

        text_to_synthesize = " ".join(sentences)
        if not text_to_synthesize:
            return
        
        tts_stream_handle = tts_deployment_handle.options(stream=True)
        gen: DeploymentResponseGenerator = tts_stream_handle.synthesize.remote(text_to_synthesize)

        async for audio_chunk in gen:
            yield audio_chunk   
