import torch
import sys
import os

# --- 1. SETUP PATHS ---
current_dir = os.getcwd()
beats_path = os.path.join(current_dir, 'unilm', 'beats')

if beats_path not in sys.path:
    sys.path.append(beats_path)

try:
    from BEATs import BEATs, BEATsConfig
except ImportError:
    print("Error: Could not import BEATs. Check your unilm/beats folder.")
    sys.exit(1)

# --- 2. LOAD LOCAL CHECKPOINT ---
# CHANGED: Point to the file you just downloaded
checkpoint_path = "./BEATs_iter3_plus_AS2M.pt" 

if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint file not found at {checkpoint_path}")
    print("Please run: wget https://huggingface.co/lpepino/beats_ckpts/resolve/main/BEATs_iter3_plus_AS2M.pt")
    sys.exit(1)

print(f"Loading Checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# --- 3. INITIALIZE MODEL ---
print("Initializing Model...")
cfg = BEATsConfig(checkpoint['cfg'])
model = BEATs(cfg)
model.load_state_dict(checkpoint['model'])

# --- 4. FREEZE MODEL ---
model.eval()
for param in model.parameters():
    param.requires_grad = False

# --- 5. EXTRACT TOKENS ---
print("Extracting features...")
input_audio = torch.randn(1, 30*16000) # 1 sec dummy audio
padding_mask = torch.zeros(1, 30*16000).bool()

with torch.no_grad():
    features = model.extract_features(input_audio, padding_mask=padding_mask)[0]

# --- 6. RESULTS --
print(features.shape)

