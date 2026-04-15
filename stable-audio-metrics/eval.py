import sys
import os
import shutil
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm

# --- 1. SETUP PATHS ---
# Assumes this script is inside the stable-audio-metrics root folder
metrics_root = os.path.dirname(os.path.abspath(__file__)) 
sys.path.insert(0, metrics_root)

from src.clap_score import clap_score
from src.passt_kld import passt_kld
from src.openl3_fd import openl3_fd

# --- CONFIGURATION ---
# INPUT: Your raw stereo generation
RAW_GENERATED_PATH = '/mnt/storage/outputs/inference_scaling_zero_order_ode_linear_6_candidates_4_search_radius_095_beats/audio'

# INPUT: The fair reference folder (886 real files)
# Created by setup_ref.py
REFERENCE_PATH = '/data/audiocaps/reference_886_fair'

# INPUT: Official CSV for text prompts
CSV_PATH = 'load/audiocaps-test.csv'

# OUTPUT: Where we save the fixed mono files
FIXED_GENERATED_PATH = '/data/audiocaps/generated_test_audios_fixed_mono'

# ---------------------------------------------------------
# STEP 1: Fix Audio Function (Transpose & Mono)
# ---------------------------------------------------------
def fix_audio_files():
    if os.path.exists(FIXED_GENERATED_PATH):
        shutil.rmtree(FIXED_GENERATED_PATH)
    os.makedirs(FIXED_GENERATED_PATH)
    
    print(f"🔧 Fixing audio shapes from {RAW_GENERATED_PATH}...")
    files = [f for f in os.listdir(RAW_GENERATED_PATH) if f.endswith('.wav')]
    
    for fname in tqdm(files):
        src = os.path.join(RAW_GENERATED_PATH, fname)
        dst = os.path.join(FIXED_GENERATED_PATH, fname)
        
        try:
            # Load
            audio, sr = sf.read(src)
            
            # FIX A: Transpose if shape is (Channels, Time) -> (2, 441000)
            if len(audio.shape) > 1 and audio.shape[0] < audio.shape[1]:
                audio = audio.T
            
            # FIX B: Force Mono (Standard for AudioCaps metrics)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=-1)
                
            # Save fixed file
            sf.write(dst, audio, sr)
            
        except Exception as e:
            print(f"⚠️ Error fixing {fname}: {e}")

    print(f"✅ Created fixed mono dataset at: {FIXED_GENERATED_PATH}")

# ---------------------------------------------------------
# EXECUTION START
# ---------------------------------------------------------

# 1. RUN THE FIX (Crucial Step!)
fix_audio_files()

# 2. PREPARE MAPPINGS
print("\n📖 Preparing ID mappings...")
df = pd.read_csv(CSV_PATH)
filename_map = {} 
valid_ids = []

# Scan our fixed folder to see what we have
available_files = set(os.listdir(FIXED_GENERATED_PATH))

for idx, row in df.iterrows():
    yt_id = row['youtube_id']
    start_time = int(row['start_time']) 
    your_filename = f"{yt_id}_{start_time}.wav"
    
    if your_filename in available_files:
        valid_ids.append(row['audiocap_id'])
        filename_map[row['audiocap_id']] = your_filename

# Create text map for CLAP
text_map = {}
for audiocap_id in valid_ids:
    row = df[df['audiocap_id'] == audiocap_id].iloc[0]
    fname = filename_map[audiocap_id]
    # CLAP expects filename without extension as key
    fname_base = os.path.splitext(fname)[0] 
    text_map[fname_base] = row['caption']

# 3. RUN METRICS

# A. CLAP SCORE
print("\nComputing CLAP score...")
clp = clap_score(text_map, FIXED_GENERATED_PATH, audio_files_extension='.wav')

# B. KL DIVERGENCE
print("\nComputing KL Divergence...")
# We use the filenames (without .wav) as IDs because we are comparing folder-to-folder
file_ids = [f[:-4] for f in os.listdir(FIXED_GENERATED_PATH) if f.endswith('.wav')]

kl = passt_kld(
    ids=file_ids, 
    eval_path=FIXED_GENERATED_PATH, 
    ref_path=REFERENCE_PATH, 
    no_ids=[], 
    collect='mean'
)

# C. FRECHET DISTANCE
print("\nComputing FD (Fixed Mono vs Fair Reference)...")
fd = openl3_fd(
    channels=1, # Explicitly tell it we are using Mono
    samplingrate=44100, 
    content_type='env', 
    openl3_hop_size=0.5,
    eval_path=FIXED_GENERATED_PATH,
    eval_files_extension='.wav',
    ref_path=REFERENCE_PATH,
    ref_files_extension='.wav',
    batching=8
)

# 4. FINAL REPORT
print("\n" + "="*50)
print("🎯 FINAL RESULTS (Fixed Mono vs Fair Reference)")
print("="*50)
print(f"CLAP Score: {clp}")
print(f"KL Divergence: {kl}")
print(f"Fréchet Distance: {fd}")
print("="*50)