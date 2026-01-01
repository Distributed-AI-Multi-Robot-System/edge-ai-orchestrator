import ray
import itertools
from stt.transcription_actor import TranscriptionActor
from stt.worker_type import WorkerType
from ray.actor import ActorProxy

@ray.remote
class TranscriptionRouter:
    """
    Centralized router that holds references to all worker actors.
    STTActors send requests here, and this router assigns them to workers.
    """
    def __init__(self, base_size=2, tiny_size=2, device="cpu"):
        # Initialize Workers
        self.main_workers = [
            TranscriptionActor.remote("base", device) for _ in range(base_size)
        ]
        self.tail_workers = [
            TranscriptionActor.remote("tiny", device) for _ in range(tiny_size)
        ]
        
        # Create cycle iterators for Round-Robin load balancing
        self.main_iter = itertools.cycle(self.main_workers)
        self.tail_iter = itertools.cycle(self.tail_workers)

    def get_worker(self, worker_type: WorkerType = WorkerType.BASE) -> ActorProxy:
        """Returns a handle to the next available worker."""
        if worker_type == WorkerType.BASE:
            return next(self.main_iter)
        return next(self.tail_iter)