import os
import soundfile as sf
import numpy as np
from tqdm import tqdm

SOURCE_DIR = '/mnt/audiocaps/reference_886_stereo'
TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_886')

if __name__ == "__main__":
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    print(f"🔧 Extracting Left channel from references in {SOURCE_DIR}...")
    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.wav')]
    
    for fname in tqdm(files):
        src_path = os.path.join(SOURCE_DIR, fname)
        dst_path = os.path.join(TARGET_DIR, fname)
        
        try:
            # Soundfile implicitly understands standard stereo structure (Time, Channels)
            audio, sr = sf.read(src_path)
            
            if len(audio.shape) > 1:
                if audio.shape[0] < audio.shape[1]:
                    audio = audio.T
                # Extract first channel and make contiguous for the C backend
                audio = np.ascontiguousarray(audio[:, 0])
                
            sf.write(dst_path, audio, sr, subtype='PCM_16')
            
        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")
            
    print(f"✅ Created mono reference dataset at: {TARGET_DIR}")
