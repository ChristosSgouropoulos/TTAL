import json
import os
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
# The JSON file defining your 886-sample subset
JSON_PATH = '/data/audiocaps/test_audiocaps_subset_updated.json'

# The folder containing the FULL AudioCaps dataset (where the real files live)
SOURCE_REF_FOLDER = '/data/audiocaps/audio'

# The new folder we will create for the "Fair" comparison
TARGET_REF_FOLDER = '/data/audiocaps/reference_886_fair'
# ---------------------

def setup_fair_reference():
    # 1. Create Target Folder
    if os.path.exists(TARGET_REF_FOLDER):
        print(f"⚠️  Cleaning existing folder: {TARGET_REF_FOLDER}")
        shutil.rmtree(TARGET_REF_FOLDER)
    os.makedirs(TARGET_REF_FOLDER)

    # 2. Load JSON Data
    print(f"📖 Reading JSON: {JSON_PATH}")
    items = []
    with open(JSON_PATH, 'r') as f:
        # Handle both standard JSON list and JSONL (line-by-line)
        content = f.read().strip()
        try:
            data = json.loads(content)
            if isinstance(data, dict) and 'test' in data:
                items = data['test']
            elif isinstance(data, list):
                items = data
            else:
                items = [data]
        except json.JSONDecodeError:
            f.seek(0)
            items = [json.loads(line) for line in f if line.strip()]

    print(f"🔍 Found {len(items)} items in JSON.")

    # 3. Link Matching Files
    count = 0
    missing = 0
    
    print("🔗 Linking reference files...")
    for item in tqdm(items):
        # Extract filename from the 'location' field (e.g., 'files/y_OyLW9lBXU_10.wav' -> 'y_OyLW9lBXU_10.wav')
        filename = os.path.basename(item['location'])
        
        # Source path (Real file)
        src = os.path.join(SOURCE_REF_FOLDER, filename)
        # Destination path (Link)
        dst = os.path.join(TARGET_REF_FOLDER, filename)
        
        if os.path.exists(src):
            # Create a symbolic link (saves space, instant)
            try:
                os.symlink(src, dst)
                count += 1
            except FileExistsError:
                pass
        else:
            missing += 1
            # Optional: print missing files to debug
            # print(f"Missing: {filename}")

    # 4. Summary
    print("-" * 40)
    print(f"✅ Successfully prepared: {TARGET_REF_FOLDER}")
    print(f"📂 Files Linked: {count}")
    print(f"❌ Files Missing: {missing}")
    print("-" * 40)
    print("You can now use this folder as 'REFERENCE_PATH' in your evaluation script.")

if __name__ == "__main__":
    setup_fair_reference()