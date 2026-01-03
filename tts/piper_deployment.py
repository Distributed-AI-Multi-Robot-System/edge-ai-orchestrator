from piper.voice import PiperVoice
from ray import serve

@serve.deployment(
    autoscaling_config={"min_replicas": 1, "max_replicas": 4}, 
    ray_actor_options={"num_cpus": 4}
)
class PiperDeployment:
    def __init__(self, model_name: str, use_cuda: bool = False):
        self.voice_model = PiperVoice.load(model_name, use_cuda=use_cuda)

    # CHANGE: Use 'def' instead of 'async def' for blocking CPU work
    def synthesize(self, text: str):
        """
        Synthesize speech from text.
        Ray Serve will run this generator in a thread, preventing event loop blocking.
        """
        for chunk in self.voice_model.synthesize(text):
            yield chunk.audio_int16_bytes