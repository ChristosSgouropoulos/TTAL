from diffusers import AutoencoderOobleck
import torch
vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    )

for param in vae.parameters():
    vae.requires_grad = False
    vae.eval()
audio_input = torch.rand(1,2,1323000)
audio_latent = vae.encode(
    audio_input
).latent_dist.sample()
audio_latent = audio_latent.transpose(
    1, 2
)
print(audio_latent.shape)