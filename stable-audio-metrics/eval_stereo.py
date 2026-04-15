import sys
import os
import shutil
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm

# --- SETUP PATHS ---
metrics_root = os.path.dirname(os.path.abspath(__file__)) 
sys.path.insert(0, metrics_root)

from src.clap_score import clap_score
from src.passt_kld import passt_kld
from src.openl3_fd import openl3_fd

# --- CONFIG ---
RAW_GENERATED_PATH = '/data/audiocaps/generated_test_audios_subset'
STEREO_REF_PATH = '/data/audiocaps/reference_886_stereo'
CSV_PATH = 'load/audiocaps-test.csv'
# Temp folder for fixed-shape (but still stereo) generation
FIXED_STEREO_GEN_PATH = '/data/audiocaps/generated_test_audios_fixed_stereo'

# 1. FIX GENERATED SHAPES (Transpose (2, N) -> (N, 2))
if os.path.exists(FIXED_STEREO_GEN_PATH):
    shutil.rmtree(FIXED_STEREO_GEN_PATH)
os.makedirs(FIXED_STEREO_GEN_PATH)

print("🔧 Preparing Generated Data (Transposing to Channels-Last Stereo)...")
for fname in tqdm(os.listdir(RAW_GENERATED_PATH)):
    if not fname.endswith('.wav'): continue
    src = os.path.join(RAW_GENERATED_PATH, fname)
    dst = os.path.join(FIXED_STEREO_GEN_PATH, fname)
    
    audio, sr = sf.read(src)
    if len(audio.shape) > 1 and audio.shape[0] < audio.shape[1]:
        audio = audio.T # Fix the (2, 441000) bug
    sf.write(dst, audio, sr)

# 2. PREPARE MAPPINGS FOR CLAP/KL
print("\n📖 Mapping IDs...")
df = pd.read_csv(CSV_PATH)
valid_ids = []
text_map = {}
available_files = set(os.listdir(FIXED_STEREO_GEN_PATH))

for _, row in df.iterrows():
    fname = f"{row['youtube_id']}_{int(row['start_time'])}.wav"
    if fname in available_files:
        valid_ids.append(row['audiocap_id'])
        text_map[fname[:-4]] = row['caption']

# 3. RUN METRICS
print("\n" + "="*50)
print("🚀 STARTING FULL STEREO EVALUATION")
print("="*50)

# CLAP (Fusion-best)
print("\n1/3. Computing CLAP Score (Stereo Input)...")
# Note: CLAP internally handles stereo by averaging channels
clp = clap_score(text_map, FIXED_STEREO_GEN_PATH, audio_files_extension='.wav')

# KL Divergence
print("\n2/3. Computing KL Divergence (886 vs 886 Stereo)...")
# Note: PaSST (KL) also averages channels to mono internally
file_ids = [f[:-4] for f in available_files]
kl = passt_kld(
    ids=file_ids, 
    eval_path=FIXED_STEREO_GEN_PATH, 
    ref_path=STEREO_REF_PATH, 
    no_ids=[], 
    collect='mean'
)

# FD (1024-dimensional comparison)
print("\n3/3. Computing FD (Channels=2, 1024-dim features)...")
fd = openl3_fd(
    channels=2, # THIS IS WHAT CHANGES THE SCORE FROM 73 TO 75
    samplingrate=44100, 
    content_type='env', 
    openl3_hop_size=0.5,
    eval_path=FIXED_STEREO_GEN_PATH,
    eval_files_extension='.wav',
    ref_path=STEREO_REF_PATH,
    ref_files_extension='.wav',
    batching=8
)

# 4. FINAL REPORT
print("\n" + "="*50)
print("🎯 OFFICIAL REPRODUCTION RESULTS (STEREO)")
print("="*50)
print(f"CLAP Score:      {clp:.4f}")
print(f"KL Divergence:   {kl:.4f}")
print(f"Fréchet Distance: {fd:.4f}")
print("="*50)