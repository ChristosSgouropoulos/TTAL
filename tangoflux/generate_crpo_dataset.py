import os
import json
import time
import torch
import argparse
import multiprocessing
from tqdm import tqdm
from safetensors.torch import load_file
from diffusers import AutoencoderOobleck
import soundfile as sf
from model import TangoFlux
import random




def generate_audio_chunk(args, chunk, gpu_id, output_dir, samplerate, return_dict, process_id):
    """
    Function to generate audio for a chunk of text prompts on a specific GPU.
    """
    try:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(device)
        print(f"Process {process_id}: Using device {device}")

        # Initialize model
        config = {
        'num_layers': 6,
        'num_single_layers': 18,
        'in_channels': 64,
        'attention_head_dim': 128,
        'joint_attention_dim': 1024,
        'num_attention_heads': 8,
        'audio_seq_len': 645,
        'max_duration': 30,
        'uncondition': False,
        'text_encoder_name': "google/flan-t5-large"
        }   

        model = TangoFlux(config)
        print(f"Process {process_id}: Loading model from {args.model} on {device}")
        w1 = load_file(args.model)
        model.load_state_dict(w1, strict=False)
        model = model.to(device)
        model.eval()

        # Initialize VAE
        vae = AutoencoderOobleck.from_pretrained("stabilityai/stable-audio-open-1.0", subfolder='vae')
        vae = vae.to(device)
        vae.eval()

        outputs = []

        # Corrected loop using enumerate properly with tqdm
        for idx, item in tqdm(enumerate(chunk), total=len(chunk), desc=f"GPU {gpu_id}"):
            text = item['captions']
            

            if os.path.exists(os.path.join(output_dir, f"id_{item['id']}_sample1.wav")):
                print("Exist! Skipping!")
                continue
            with torch.no_grad():
                latent = model.inference_flow(
                    text,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    duration=10,
                    num_samples_per_prompt=args.num_samples
                )
               
                #waveform_end = int(duration * vae.config.sampling_rate)
                latent = latent[:, :220, :]   ## 220 correspond to the latent length of audiocaps encoded with this vae. You can modify this 
                wave = vae.decode(latent.transpose(2, 1)).sample.cpu()

                for i in range(args.num_samples):
                    filename = f"id_{item['id']}_sample{i+1}.wav"
                    filepath = os.path.join(output_dir, filename)

                    sf.write(filepath, wave[i].T, samplerate)
                    outputs.append({
                        "id": item['id'],
                        "sample": i + 1,
                        "path": filepath,
                        "captions": text
                    })

        return_dict[process_id] = outputs
        print(f"Process {process_id}: Completed processing on GPU {gpu_id}")

    except Exception as e:
        print(f"Process {process_id}: Error on GPU {gpu_id}: {e}")
        return_dict[process_id] = []

def split_into_chunks(data, num_chunks):
    """
    Splits data into num_chunks approximately equal parts.
    """
    avg = len(data) // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * avg
        # Ensure the last chunk takes the remainder
        end = (i + 1) * avg if i != num_chunks - 1 else len(data)
        chunks.append(data[start:end])
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Generate audio using multiple GPUs")
    parser.add_argument('--num_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--model', type=str, required=True, help='Path to tangoflux weights')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples per prompt')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--json_path', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--sample_size', type=int, default=20000, help='Number of prompts to sample for CRPO')
    parser.add_argument('--guidance_scale', type=float, default=4.5, help='Guidance scale used for generation')
    args = parser.parse_args()

    # Check GPU availability 
    num_gpus = torch.cuda.device_count()
    sample_size = args.sample_size
    

    # Load JSON data
    import json
    try:
        with open(args.json_path, 'r') as f:
            data = json.load(f)
        
    except Exception as e:
        print(f"Error loading JSON file {args.json_path}: {e}")
        return

    if not isinstance(data, list):
        print("Error: JSON data is not a list.")
        return

    if len(data) < sample_size:
        print(f"Warning: JSON data contains only {len(data)} items. Sampling all available data.")
        sampled = data
    else:
        sampled = random.sample(data, sample_size)

    # Split data into chunks based on available GPUs
    random.shuffle(sampled)
    chunks = split_into_chunks(sampled, num_gpus)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    samplerate = 44100

    # Manager for inter-process communication
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(
            target=generate_audio_chunk,
            args=(
                args,
                chunks[i],
                i,  # GPU ID
                args.output_dir,
                samplerate,
                return_dict,
                i,  # Process ID
               
            )
        )
        processes.append(p)
        p.start()
        print(f"Started process {i} on GPU {i}")

    for p in processes:
        p.join()
        print(f"Process {p.pid} has finished.")

    # Aggregate results
    
    

    


    audio_info_list = [
        [{
            "path": f"{args.output_dir}/id_{sampled[j]['id']}_sample{i}.wav",
            "duration": sampled[j]["duration"],
            "captions": sampled[j]["captions"]
        }
        for i in range(1, args.num_samples+1) ] for j in range(sample_size)
    ]

    #print(audio_info_list)

    with open(f'{args.output_dir}/results.json','w') as f:
        json.dump(audio_info_list,f)
        
    print(f"All audio samples have been generated and saved to {args.output_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()