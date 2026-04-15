from huggingface_hub import snapshot_download
import os

print("--- Starting Download from Hugging Face ---")
print("This will download approx 60GB. Please wait...")

# This function is the python equivalent of the CLI command
snapshot_download(
    repo_id="Zhaowc/AudioCaps",
    repo_type="dataset",
    local_dir="/data/AudioCaps_HF",
    local_dir_use_symlinks=False  # Forces actual file download, not links
)

print("--- Download Complete! ---")