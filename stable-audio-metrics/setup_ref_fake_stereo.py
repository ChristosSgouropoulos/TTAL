import os
import numpy as np
import soundfile as sf
from tqdm import tqdm

# --- CONFIG ---
# Folder containing the 886 mono reference files
MONO_REF_PATH = '/data/audiocaps/reference_886_fair'
# New folder for the pseudo-stereo versions
STEREO_REF_PATH = '/data/audiocaps/reference_886_stereo'
# --------------

if not os.path.exists(STEREO_REF_PATH):
    os.makedirs(STEREO_REF_PATH)

print(f"🔊 Converting mono reference to pseudo-stereo (Channel Duplication)...")
files = [f for f in os.listdir(MONO_REF_PATH) if f.endswith('.wav')]

for fname in tqdm(files):
    src = os.path.join(MONO_REF_PATH, fname)
    dst = os.path.join(STEREO_REF_PATH, fname)
    
    # Load mono
    audio, sr = sf.read(src)
    
    # If it's already stereo (some datasets are inconsistent), just copy
    if len(audio.shape) > 1 and audio.shape[1] == 2:
        sf.write(dst, audio, sr)
    else:
        # Duplicate the channel: (N,) -> (N, 2)
        stereo_audio = np.stack([audio, audio], axis=-1)
        sf.write(dst, stereo_audio, sr)

print(f"✅ Created stereo reference at: {STEREO_REF_PATH}")