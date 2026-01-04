import logging
import os
from ray import serve
from tts.piper_deployment import PiperDeployment

from tts.tts_actor import TTSActor
logger = logging.getLogger(__name__)

PIPER_BASE_DEPLOYMENT_NAME = "piper_tts_deployment"
PIPER_DEPLOYMENT_NAME_EN = f"{PIPER_BASE_DEPLOYMENT_NAME}_en"
PIPER_DEPLOYMENT_NAME_DE = f"{PIPER_BASE_DEPLOYMENT_NAME}_de"
PIPER_DEPLOYMENT_NAME_IT = f"{PIPER_BASE_DEPLOYMENT_NAME}_it"
PIPER_DEPLOYMENT_NAME_FR = f"{PIPER_BASE_DEPLOYMENT_NAME}_fr"


class TTSManager:
    """
    Manages Ray-based TTS infrastructure.
    
    - Creates and manages Text-to-SpeechActor deployments
    - Spawns TTSActor per WebSocket session
    """

    def start(self):
        logger.info("Starting Ray Serve Deployments...")
        # It is intended to use that much CPU for the TTS deployments, we do here an overprovisioning since not all deployments are used by one actor at the same time
        piper_deployment_english = PiperDeployment.options(
            name=PIPER_DEPLOYMENT_NAME_EN, 
            ray_actor_options={"num_cpus": 1}
        ).bind(model_path="tts/models/en_US-lessac-high.onnx")
        
        piper_deployment_german = PiperDeployment.options(
            name=PIPER_DEPLOYMENT_NAME_DE, 
            ray_actor_options={"num_cpus": 1}
        ).bind(model_path="tts/models/de_DE-thorsten-high.onnx")

        piper_deployment_italian = PiperDeployment.options(
            name=PIPER_DEPLOYMENT_NAME_IT, 
            ray_actor_options={"num_cpus": 1}
        ).bind(model_path="tts/models/it_IT-paola-medium.onnx")

        piper_deployment_french = PiperDeployment.options(
            name=PIPER_DEPLOYMENT_NAME_FR, 
            ray_actor_options={"num_cpus": 1}
        ).bind(model_path="tts/models/fr_FR-siwis-medium.onnx")

        serve.run(piper_deployment_english, name=PIPER_DEPLOYMENT_NAME_EN, route_prefix="/en")
        serve.run(piper_deployment_german, name=PIPER_DEPLOYMENT_NAME_DE, route_prefix="/de")
        serve.run(piper_deployment_italian, name=PIPER_DEPLOYMENT_NAME_IT, route_prefix="/it")
        serve.run(piper_deployment_french, name=PIPER_DEPLOYMENT_NAME_FR, route_prefix="/fr")
        
        logger.info("Ray Serve Deployments Ready: 'WhisperBase' and 'WhisperTiny'")


    def register(self, session_id: str):
        """
        Register a new session and spawn its STTActor.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            STTActor handle for the session
        """
        name  = f"tts_actor_{session_id}"
        actor = TTSActor.options(name=name, get_if_exists=True).remote()
        logger.debug(f"Session {session_id} registered")
        
        return actor
    
    def get_deployment_handle(self, language: str):
            """Returns the correct DeploymentHandle for the requested language."""
            match language:
                case "en":
                    name = PIPER_DEPLOYMENT_NAME_EN
                case "de":
                    name = PIPER_DEPLOYMENT_NAME_DE
                case "it":
                    name = PIPER_DEPLOYMENT_NAME_IT
                case "fr":
                    name = PIPER_DEPLOYMENT_NAME_FR
                case _:
                    # Fallback to English or raise error
                    logger.warning(f"Unsupported language {language}, falling back to English")
                    name = PIPER_DEPLOYMENT_NAME_EN
                
            return serve.get_app_handle(name)
