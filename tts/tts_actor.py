import ray
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator


@ray.remote
class TTSActor:
    def __init__(self, tts_deployment_handle: DeploymentHandle):
        self.text_buffer = ""
        self.tts_stream_handle = tts_deployment_handle.options(stream=True)
        print("[TTSActor] Initialized and ready")

    async def synthesize_text(self, text: str):
        """
        Synthesize text to speech using the PiperDeployment.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            Async generator yielding audio chunks as bytes
        """
        gen: DeploymentResponseGenerator = (
            await self.tts_stream_handle.synthesize.remote(text)
        )

        async for audio_chunk in gen:
            yield audio_chunk
