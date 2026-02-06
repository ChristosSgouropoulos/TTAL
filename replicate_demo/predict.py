# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import json
from cog import BasePredictor, Input, Path
from diffusers import AutoencoderOobleck
import soundfile as sf
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from tangoflux.model import TangoFlux
from tangoflux import TangoFluxInference

MODEL_CACHE = "model_cache"
MODEL_URL = (
    "https://weights.replicate.delivery/default/declare-lab/TangoFlux/model_cache.tar"
)


class CachedTangoFluxInference(TangoFluxInference):
    ## load the weights from replicate.delivery for faster booting
    def __init__(self, name="declare-lab/TangoFlux", device="cuda", cached_paths=None):
        if cached_paths:
            paths = cached_paths
        else:
            paths = snapshot_download(repo_id=name)

        self.vae = AutoencoderOobleck()
        vae_weights = load_file(f"{paths}/vae.safetensors")
        self.vae.load_state_dict(vae_weights)
        weights = load_file(f"{paths}/tangoflux.safetensors")

        with open(f"{paths}/config.json", "r") as f:
            config = json.load(f)
        self.model = TangoFlux(config)
        self.model.load_state_dict(weights, strict=False)
        self.vae.to(device)
        self.model.to(device)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            print("downloading")
            download_weights(MODEL_URL, MODEL_CACHE)

        self.model = CachedTangoFluxInference(
            cached_paths=f"{MODEL_CACHE}/declare-lab/TangoFlux"
        )

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt", default="Hammer slowly hitting the wooden table"
        ),
        duration: int = Input(
            description="Duration of the output audio in seconds", default=10
        ),
        steps: int = Input(
            description="Number of inference steps", ge=1, le=200, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=4.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        audio = self.model.generate(
            prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            duration=duration,
        )
        audio_numpy = audio.numpy()
        out_path = "/tmp/out.wav"

        sf.write(
            out_path, audio_numpy.T, samplerate=self.model.vae.config.sampling_rate
        )
        return Path(out_path)
