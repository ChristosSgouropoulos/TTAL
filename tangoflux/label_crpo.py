import os
import json
import argparse
import torch
import laion_clap
import numpy as np
import multiprocessing
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Labelling clap score for crpo dataset"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of audio samples per prompt"
    )
    parser.add_argument(
        "--json_path", type=str, required=True,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the final JSON with CLAP scores"
    )
    return parser.parse_args()

#python3 label_clap.py --json_path=/mnt/data/chiayu/crpo/crpo_iteration1/results.json --output_dir=/mnt/data/chiayu/crpo/crpo_iteration1
@torch.no_grad()
def compute_clap(model, audio_files, text_data):
    # Compute audio and text embeddings, then compute the dot product (CLAP score)
    audio_embed = model.get_audio_embedding_from_filelist(x=audio_files, use_tensor=True)
    text_embed = model.get_text_embedding(text_data, use_tensor=True)
    return audio_embed @ text_embed.T

def process_chunk(args, chunk, gpu_id, return_dict, process_id):
    """
    Process a chunk of the data on a specific GPU.
    Loads the CLAP model on the designated device, then for each item in the chunk,
    computes the CLAP scores and attaches them to the data.
    """
    try:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(device)
        print(f"Process {process_id}: Using device {device}")

        # Initialize the CLAP model on this GPU
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.to(device)
        model.load_ckpt()
        model.eval()

        for j, item in enumerate(tqdm(chunk, desc=f"GPU {gpu_id}")):
            # Each item is assumed to be a list of samples.
            # Skip if already computed.
            if 'clap_score' in item[0]:
                continue

            # Collect audio file paths and text data (using the first caption)
            audio_files = [item[i]['path'] for i in range(args.num_samples)]
            text_data = [item[0]['captions']]

            try:
                clap_scores = compute_clap(model, audio_files, text_data)
            except Exception as e:
                print(f"Error processing item index {j} on GPU {gpu_id}: {e}")
                continue

            # Attach the computed score to each sample in the item
            for k in range(args.num_samples):
                item[k]['clap_score'] = np.round(clap_scores[k].item(), 3)

        return_dict[process_id] = chunk
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
        # Ensure the last chunk takes the remainder of the data
        end = (i + 1) * avg if i != num_chunks - 1 else len(data)
        chunks.append(data[start:end])
    return chunks

def main():
    args = parse_args()

    # Load data from JSON and slice by start/end if provided
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    # Check GPU availability and split data accordingly
    num_gpus = torch.cuda.device_count()

    print(f"Found {num_gpus} GPUs. Splitting data into {num_gpus} chunks.")
    chunks = split_into_chunks(data, num_gpus)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a manager dict to collect results from all processes
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    for i in range(num_gpus):
        p = multiprocessing.Process(
            target=process_chunk,
            args=(args, chunks[i], i, return_dict, i)
        )
        processes.append(p)
        p.start()
        print(f"Started process {i} on GPU {i}")

    for p in processes:
        p.join()
        print(f"Process {p.pid} has finished.")

    # Aggregate all chunks back into a single list
    combined_data = []
    for i in range(num_gpus):
        combined_data.extend(return_dict[i])

    # Save the combined results to a single JSON file
    output_file =  f"{args.output_dir}/clap_scores.json"
    with open(output_file, 'w') as f:
        json.dump(combined_data, f)
    print(f"All CLAP scores have been computed and saved to {output_file}")

    max_item = [max(x, key=lambda item: item['clap_score']) for x in combined_data]
    min_item = [min(x, key=lambda item: item['clap_score']) for x in combined_data]

    crpo_dataset = []
    for chosen,reject in zip(max_item,min_item):
        crpo_dataset.append({"captions": chosen['captions'], 
        "duration": chosen['duration'], 
        "chosen": chosen['path'], 
        "reject": reject['path']})
        
    with open(f"{args.output_dir}/train.json",'w') as f:
        json.dump(crpo_dataset,f)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
