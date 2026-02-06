from setuptools import setup

setup(
    name="tangoflux",
    description="TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching",
    version="0.1.0",
    packages=["tangoflux"],
    install_requires=[
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "torchlibrosa==0.1.0",
        "torchvision==0.19.0",
        "transformers==4.44.0",
        "diffusers==0.30.0",
        "accelerate==0.34.2",
        "datasets==2.21.0",
        "librosa",
        "tqdm",
        "wandb",
        "click",
        "gradio",
        "torchaudio",
    ],
    entry_points={
        "console_scripts": [
            "tangoflux=tangoflux.cli:main",
            "tangoflux-demo=tangoflux.demo:main",
        ],
    },
)
