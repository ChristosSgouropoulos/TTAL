import os
import pandas as pd
import subprocess
import time          # <--- Added
import random        # <--- Added
from tqdm import tqdm

# === CONFIGURATION ===
# Where to save the files. 
DATA_ROOT = "/data/AUDIOCAPS/audio_32000Hz"
COOKIES_FILE = "cookies.txt" 

# Metadata URLs
URLS = {
    "train": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv",
    # "val": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/val.csv",
    # "test": "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv"
}

def download_subset(subset, csv_url):
    print(f"\n=== Processing {subset} set ===")
    
    subset_dir = os.path.join(DATA_ROOT, subset)
    os.makedirs(subset_dir, exist_ok=True)
    
    df = pd.read_csv(csv_url)
    print(f"Total files to process: {len(df)}")
    
    success_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        yid = row['youtube_id']
        start_time = row['start_time']
        end_time = start_time + 10.0
        
        output_path = os.path.join(subset_dir, f"{yid}.wav")
        
        # 1. Skip if exists (Efficiency)
        if os.path.exists(output_path):
            success_count += 1
            continue

        # Construct the Modern yt-dlp command
        cmd = [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "--force-overwrites",
            "--download-sections", f"*{start_time}-{end_time}",
            "--remote-components", "ejs:github",
            "--cookies", COOKIES_FILE,
            "-o", f"{subset_dir}/{yid}.%(ext)s",
            f"https://www.youtube.com/watch?v={yid}"
        ]
        
        try:
            # capture_output=True keeps the terminal clean.
            # If it gets stuck, change to capture_output=False to see errors.
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if os.path.exists(output_path):
                success_count += 1
          
        except Exception as e:
            pass 

    print(f"Finished {subset}. Total files on disk: {success_count}/{len(df)}")

if __name__ == "__main__":
    if not os.path.exists(COOKIES_FILE):
        print("⚠️  WARNING: cookies.txt not found! YouTube may block downloads.")
    
    # Only running train since you finished the others
    download_subset("train", URLS["train"])