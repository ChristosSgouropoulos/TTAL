#!/usr/bin/env python3
"""
Evaluation script for TangoFlux audio generation.

Usage:
    # Run with default config:
    python eval_updated.py

    # Run with custom config:
    python eval_updated.py --config eval_config.json

    # Override specific paths from CLI:
    python eval_updated.py --config eval_config.json --raw_generated_path /path/to/audio
"""

import sys
import os
import shutil
import argparse
import numpy as np
import soundfile as sf
import json
import subprocess
from tqdm import tqdm

# --- SETUP PATHS ---
metrics_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, metrics_root)

from src.clap_score import clap_score
from src.passt_kld import passt_kld
from src.openl3_fd import openl3_fd


def load_config(config_path):
    """Load JSON config file."""
    with open(config_path, "r") as f:
        return json.load(f)


def build_parser():
    p = argparse.ArgumentParser(description="TangoFlux Evaluation Metrics")
    p.add_argument("--config", type=str, default=os.path.join(metrics_root, "eval_config.json"),
                   help="Path to JSON config file (default: eval_config.json)")
    p.add_argument("--raw_generated_path", type=str, default=None,
                   help="Path to raw generated audio files")
    p.add_argument("--reference_path", type=str, default=None,
                   help="Path to reference audio files")
    p.add_argument("--json_path", type=str, default=None,
                   help="Path to AudioCaps JSON with captions")
    p.add_argument("--fixed_generated_path", type=str, default=None,
                   help="Path for preprocessed (mono, 10s) audio output")
    p.add_argument("--target_sr", type=int, default=None)
    p.add_argument("--target_duration_sec", type=int, default=None)
    return p


def resolve_path(path, base_dir):
    """Resolve a path: if relative, make it relative to base_dir."""
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def prepare_data(raw_path, fixed_path, target_duration_sec):
    if os.path.exists(fixed_path):
        shutil.rmtree(fixed_path)
    os.makedirs(fixed_path)

    print(f"🔧 Fixing audio shapes & truncating to {target_duration_sec}s from {raw_path}...")
    files = [f for f in os.listdir(raw_path) if f.endswith('.wav')]

    for fname in tqdm(files):
        src = os.path.join(raw_path, fname)
        dst = os.path.join(fixed_path, fname)

        try:
            audio, sr = sf.read(src)

            # FIX A: Transpose if shape is (Channels, Time)
            if len(audio.shape) > 1 and audio.shape[0] < audio.shape[1]:
                audio = audio.T

            # FIX B: Force Mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=-1)

            # FIX C: Truncate to target duration
            max_length = target_duration_sec * sr
            if audio.shape[0] > max_length:
                audio = audio[:max_length]

            sf.write(dst, audio, sr)

        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")

    print(f"✅ Created truncated {target_duration_sec}s mono dataset at: {fixed_path}")


def build_text_map(json_path, fixed_path):
    """Build caption mapping from AudioCaps JSON."""
    text_map = {}
    if not os.path.exists(json_path):
        print(f"⚠️ Warning: json_path '{json_path}' not found.")
        return text_map

    with open(json_path, 'r') as f:
        data = json.load(f)

    available_files = set(os.listdir(fixed_path))
    for item in data:
        basename = os.path.splitext(os.path.basename(item['location']))[0]
        wav_name = basename + '.wav'
        if wav_name in available_files:
            text_map[basename] = item['captions']
    print(f"Mapped {len(text_map)} captions successfully (out of {len(data)} in JSON).")
    return text_map


def run_eval(cfg):
    raw_generated_path = cfg["raw_generated_path"]
    reference_path = resolve_path(cfg.get("reference_path", "reference_886"), metrics_root)
    json_path = cfg["json_path"]
    fixed_generated_path = resolve_path(cfg.get("fixed_generated_path", "test_generated_mean_10s"), metrics_root)
    target_sr = cfg.get("target_sr", 44100)
    target_duration_sec = cfg.get("target_duration_sec", 10)
    metrics = cfg.get("metrics", ["clap", "kld", "fd"])

    # Print config
    print("=" * 50)
    print("  TangoFlux Evaluation")
    print("=" * 50)
    print(f"  Raw audio:    {raw_generated_path}")
    print(f"  Reference:    {reference_path}")
    print(f"  JSON:         {json_path}")
    print(f"  Output:       {fixed_generated_path}")
    print(f"  Duration:     {target_duration_sec}s")
    print(f"  Metrics:      {metrics}")
    print("=" * 50)

    # 1. Preprocess
    prepare_data(raw_generated_path, fixed_generated_path, target_duration_sec)

    # 2. Build text map
    print("\n📖 Preparing ID mappings from JSON...")
    text_map = build_text_map(json_path, fixed_generated_path)

    # 3. Run metrics
    results = {}

    if "clap" in metrics:
        print("\n▶ Computing CLAP score...")
        if text_map:
            results["clap"] = clap_score(text_map, fixed_generated_path, audio_files_extension='.wav')
        else:
            results["clap"] = "N/A (JSON missing)"
            print("Skipped CLAP because text_map could not be built from JSON.")

    if "kld" in metrics:
        print("\n▶ Computing KL Divergence (PaSST)...")
        file_ids = [f[:-4] for f in os.listdir(fixed_generated_path) if f.endswith('.wav')]
        results["kld"] = passt_kld(
            ids=file_ids,
            eval_path=fixed_generated_path,
            ref_path=reference_path,
            no_ids=[],
            collect='mean'
        )

    if "fd" in metrics:
        print("\n▶ Computing Fréchet Distance (OpenL3)...")
        results["fd"] = openl3_fd(
            channels=1,
            samplingrate=target_sr,
            content_type='env',
            openl3_hop_size=0.5,
            eval_path=fixed_generated_path,
            eval_files_extension='.wav',
            ref_path=reference_path,
            ref_files_extension='.wav',
            batching=8
        )

    if "is" in metrics:
        print("\n▶ Computing Inception Score (AudioLDM Eval)...")
        try:
            result = subprocess.run([
                "audioldm_eval",
                "-a", fixed_generated_path,
                "--metrics", "is"
            ], capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("[Warnings]", result.stderr)
            results["is"] = "See stdout above"
        except Exception as e:
            print(f"⚠️ audioldm_eval not available: {e}")
            results["is"] = "N/A"

    # 4. Final report
    print("\n" + "=" * 50)
    print("🎯 EVALUATION RESULTS")
    print("=" * 50)
    for metric, value in results.items():
        print(f"  {metric.upper():20s} {value}")
    print("=" * 50)

    # 5. Cleanup temporary preprocessed files
    if os.path.exists(fixed_generated_path):
        shutil.rmtree(fixed_generated_path)
        print(f"\n🧹 Cleaned up temporary directory: {fixed_generated_path}")

    return results


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        cfg = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        print(f"[WARNING] Config '{args.config}' not found, using defaults.")
        cfg = {}

    # CLI overrides
    for key in ["raw_generated_path", "reference_path", "json_path",
                "fixed_generated_path", "target_sr", "target_duration_sec"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    run_eval(cfg)
