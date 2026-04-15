import os
import soundfile as sf
import numpy as np
from tqdm import tqdm

src_dir = '/mnt/storage/audiocaps/reference_886_stereo'
dst_dir = '/mnt/storage/audiocaps/reference_886_mono'

os.makedirs(dst_dir, exist_ok=True)
files = [f for f in os.listdir(src_dir) if f.endswith('.wav')]

print(f"Converting {len(files)} files to Mono...")
mismatch_count = 0

for f in tqdm(files):
    src_path = os.path.join(src_dir, f)
    dst_path = os.path.join(dst_dir, f)
    
    audio, sr = sf.read(src_path)
    
    if len(audio.shape) > 1 and audio.shape[1] == 2:
        # Check if identical
        diff = np.abs(audio[:, 0] - audio[:, 1]).max()
        if diff > 1e-6:
            mismatch_count += 1
            # If not perfectly identical, average them
            audio_mono = np.mean(audio, axis=1)
        else:
            # If identical, just take the first channel
            audio_mono = audio[:, 0]
    else:
        audio_mono = audio
        
    sf.write(dst_path, audio_mono, sr)

print(f"Done! Saved to {dst_dir}")
if mismatch_count > 0:
    print(f"Note: {mismatch_count} files had differences between Left and Right, so they were averaged.")
else:
    print("All stereo files had perfectly identical Left and Right channels, so the first channel was extracted.")
