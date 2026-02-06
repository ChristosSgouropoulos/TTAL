import os
import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import random
from torch.utils.data import Dataset

class TangoFluxDataset(Dataset):
    def __init__(self, csv_path, audio_dir, target_sr=44100, target_duration=30.0):
        """
        Args:
            csv_path (str): Path to train.csv, test.csv, or val.csv
            audio_dir (str): Path to the folder containing wav files (/data/audiocaps/audio)
            target_sr (int): 44100 Hz
            target_duration (float): 30.0 seconds
        """
        self.audio_dir = audio_dir
        self.target_sr = target_sr
        self.target_samples = int(target_sr * target_duration)
        
        # 1. Load CSV
        # We assume headers: audiocap_id, youtube_id, start_time, caption
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # 2. Filter & Verify Files
        # We need to map the CSV row to a filename. 
        # Strategy: Try 'youtube_id.wav' first.
        self.data = []
        missing_count = 0
        
        for _, row in self.df.iterrows():
            yt_id = str(row['youtube_id'])
            caption = row['caption']
            
            # Construct potential filenames
            # (Users' datasets vary: sometimes 'id.wav', sometimes 'id_start.wav')
            potential_filenames = [
                f"{yt_id}.wav",
                f"{row['audiocap_id']}.wav",
                f"{yt_id}_{row['start_time']}.wav"
            ]
            
            found_path = None
            for fname in potential_filenames:
                full_path = os.path.join(audio_dir, fname)
                if os.path.exists(full_path):
                    found_path = full_path
                    break
            
            if found_path:
                self.data.append({
                    "path": found_path,
                    "caption": caption
                })
            else:
                missing_count += 1

        print(f"Loaded {len(self.data)} items.")
        if missing_count > 0:
            print(f"Warning: Could not find audio files for {missing_count} rows.")

    def __len__(self):
        return len(self.data)

    def preprocess_audio(self, audio_path):
        """
        Reads raw audio and converts to TangoFlux Input: [2, 1323000]
        """
        # A. LOAD
        # Load audio (torchaudio normalizes to [-1, 1])
        waveform, sr = torchaudio.load(audio_path)

        # B. RESAMPLE (32k -> 44.1k)
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # C. STEREO DUPLICATION (The "Pseudo-Stereo" Fix)
        # Fixes compatibility with Stable Audio VAE
        if waveform.shape[0] == 1:
            waveform = torch.cat([waveform, waveform], dim=0)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :] # Discard extra channels

        # D. DURATION (Strict 30s)
        current_samples = waveform.shape[1]

        if current_samples > self.target_samples:
            # CENTER CROP
            start = (current_samples - self.target_samples) // 2
            waveform = waveform[:, start : start + self.target_samples]
            
        elif current_samples < self.target_samples:
            # PAD WITH SILENCE (Right Side)
            pad_amount = self.target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item['path']
        caption = item['caption']

        try:
            # 1. Get processed tensor
            audio_tensor = self.preprocess_audio(audio_path)
            
            # 2. Return Dict
            return {
                "audio": audio_tensor,  # Shape: [2, 1323000]
                "text": caption         # Raw String
            }
            
        except Exception as e:
            print(f"Corrupt file {audio_path}: {e}")
            # Robust fallback: pick a random item to avoid crashing the batch
            return self.__getitem__(random.randint(0, len(self.data)-1))

# --- QUICK TEST BLOCK ---
if __name__ == "__main__":
    # Settings"
    train_dataset = TangoFluxDataset(csv_path = "/data/audiocaps/train.csv", audio_dir = "/data/audiocaps/audio")
    val_dataset = TangoFluxDataset(csv_path = "/data/audiocaps/val.csv", audio_dir = "/data/audiocaps/audio")
    test_dataset = TangoFluxDataset(csv_path = "/data/audiocaps/test.csv", audio_dir = "/data/audiocaps/audio")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

