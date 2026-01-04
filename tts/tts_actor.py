import ray
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator


@ray.remote
class TTSActor:
    def __init__(self):
        self.text_buffer = ""
        print("[TTSActor] Initialized and ready")

    async def synthesize_text(self, text: str, tts_deployment_handle: DeploymentHandle):
        """
        Synthesize text to speech using the PiperDeployment.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            Async generator yielding audio chunks as bytes
        """
        tts_stream_handle = tts_deployment_handle.options(stream=True)
        gen: DeploymentResponseGenerator = tts_stream_handle.synthesize.remote(text)

        async for audio_chunk in gen:
            yield audio_chunk   
