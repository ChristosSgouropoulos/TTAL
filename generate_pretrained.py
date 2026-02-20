import os
import torch
import torchaudio
from tangoflux import TangoFluxInference
from datasets import load_dataset
from tqdm import tqdm

def generate_all_aggregations(
    input_json_path, 
    base_output_dir, 
    steps=50, 
    duration=10, 
    device_id="0"
):
    """
    Generates audio and saves four versions (left, right, mean, stereo) 
    into their respective subfolders.
    """
    # 1. Environment and Path Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define and create the 4 required subfolders
    modes = ["left", "right", "mean", "stereo"]
    paths = {mode: os.path.join(base_output_dir, mode) for mode in modes}
    
    for folder_path in paths.values():
        os.makedirs(folder_path, exist_ok=True)

    # 2. Initialize Model
    print(f"Initializing TangoFlux on {device}...")
    model = TangoFluxInference(name='declare-lab/TangoFlux', device=device)

    # 3. Load Dataset
    dataset = load_dataset("json", data_files={"test": input_json_path}, split="test")
    print(f"Loaded {len(dataset)} prompts.")

    # 4. Inference Loop
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Processing Batches"):
        prompt = item.get('captions')
        original_name = os.path.basename(item.get('location'))

        try:
            # Generate raw stereo audio [2, Samples]
            stereo_audio = model.generate(prompt, steps=steps, duration=duration)
            
            # Ensure it is on CPU for processing and saving
            if stereo_audio.is_cuda:
                stereo_audio = stereo_audio.cpu()

            # Handle edge case where model might return mono
            if stereo_audio.shape[0] < 2:
                print(f"Warning: Sample {i} returned mono. Copying to all folders.")
                for mode in modes:
                    torchaudio.save(os.path.join(paths[mode], original_name), stereo_audio, 44100)
                continue

            # 5. Create the 4 variations
            versions = {
                "stereo": stereo_audio,
                "left":   stereo_audio[0:1, :],
                "right":  stereo_audio[1:2, :],
                "mean":   torch.mean(stereo_audio, dim=0, keepdim=True)
            }

            # 6. Save each version
            for mode, audio_tensor in versions.items():
                save_path = os.path.join(paths[mode], original_name)
                torchaudio.save(save_path, audio_tensor, 44100)
            
        except Exception as e:
            print(f"\n[Error] Failed sample {i}: {e}")

    print(f"\nSuccess! All versions saved to: {base_output_dir}")

# --- Execute ---
if __name__ == "__main__":
    generate_all_aggregations(
        input_json_path='/data/audiocaps/test_audiocaps.json',
        base_output_dir='/data/audiocaps/test_generated',
        steps=50,
        duration=10,
        device_id="0"
    )