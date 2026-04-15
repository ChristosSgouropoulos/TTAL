import os
import json

raw_gen = '/mnt/audiocaps/test_generated/mean'
fix_gen = 'test_generated_mean_10s'
ref_path = 'reference_886'
json_path = '/mnt/audiocaps/test_audiocaps.json'

print("--- 📊 Evaluation Count Check ---")
if os.path.exists(raw_gen):
    count = len([f for f in os.listdir(raw_gen) if f.endswith('.wav')])
    print(f"1. Files in Raw Generated Directory: {count}")
else:
    print("1. Raw Generated Directory: NOT FOUND")

if os.path.exists(fix_gen):
    count = len([f for f in os.listdir(fix_gen) if f.endswith('.wav')])
    print(f"2. Files in Fixed (10s) Directory: {count}")
else:
    print("2. Fixed Directory: NOT FOUND")

if os.path.exists(ref_path):
    count = len([f for f in os.listdir(ref_path) if f.endswith('.wav')])
    print(f"3. Files in Reference Directory: {count}")
else:
    print("3. Reference Directory: NOT FOUND")

if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"4. Total Records in JSON Caption Map: {len(data)}")
    
    if os.path.exists(fix_gen):
        available = set(os.listdir(fix_gen))
        mapped = 0
        for item in data:
            bn = os.path.splitext(os.path.basename(item['location']))[0] + '.wav'
            if bn in available:
                mapped += 1
        print(f"5. Successfully Mapped to Captions (for CLAP): {mapped}")
else:
    print("4. JSON Map: NOT FOUND")
