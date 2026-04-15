#!/usr/bin/env python3
"""
Main entry point for Inference-Time Scaling on TangoFlux.

Usage:
    # Run with config file:
    python main_inference_scaling.py --config inference_scaling.json

    # Override specific args from CLI:
    python main_inference_scaling.py --config inference_scaling.json \
        --scaling_method particle_filter --n_particles 4 --sample_method sde

    # Limit to a few samples for debugging:
    python main_inference_scaling.py --config inference_scaling.json --max_samples 3
"""

import os
import sys
import json
import time
import argparse
import random
import traceback

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from tangoflux import TangoFluxInference
from tangoflux_prior import TangoFluxPrior
from verifiers import EnsembleAudioReward
from inference_scaling_algorithms import SCALING_METHODS


def load_config(config_path):
    """Load JSON config file."""
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg


def merge_cli_overrides(cfg, args):
    """Merge CLI arguments into config (CLI takes priority)."""
    cli_dict = vars(args)
    for key, value in cli_dict.items():
        if key == "config":
            continue
        if value is not None:
            # Handle nested reward_weights
            if key.startswith("reward_w_"):
                reward_key = key.replace("reward_w_", "")
                if "reward_weights" not in cfg:
                    cfg["reward_weights"] = {}
                cfg["reward_weights"][reward_key] = value
            else:
                cfg[key] = value
    return cfg


def build_parser():
    """Build argument parser with all configurable parameters."""
    p = argparse.ArgumentParser(description="Inference-Time Scaling for TangoFlux")

    # Config file
    p.add_argument("--config", type=str, default="inference_scaling.json",
                   help="Path to JSON config file")

    # Model & data
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--test_set_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="Device for TangoFlux model (default: cuda:0)")
    p.add_argument("--reward_device", type=str, default=None,
                   help="Device for reward models (default: cuda:1). Use 'cpu' if single GPU.")
    p.add_argument("--seed", type=int, default=None)

    # Audio generation
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--sample_rate", type=int, default=None)

    # Scaling strategy
    p.add_argument("--scaling_method", type=str, default=None,
                   choices=["bon", "particle_filter", "rbf", "dps", "zero_order"],
                   help="Inference-time scaling method")
    p.add_argument("--sample_method", type=str, default=None,
                   choices=["ode", "sde"],
                   help="ODE (deterministic) or SDE (stochastic) sampling")
    p.add_argument("--interpolant", type=str, default=None,
                   choices=["linear", "vp"],
                   help="Flow interpolant: linear or variance-preserving")

    # Method-specific params
    p.add_argument("--n_particles", type=int, default=None,
                   help="Number of particles/samples (BoN=n_samples, PF/RBF=n_particles)")
    p.add_argument("--max_nfe", type=int, default=None,
                   help="Max NFE budget (RBF)")
    p.add_argument("--temperature", type=float, default=None,
                   help="Softmax temperature for particle resampling")
    p.add_argument("--eval_interval", type=int, default=None,
                   help="Evaluate rewards every K steps")
    p.add_argument("--guidance_strength", type=float, default=None,
                   help="DPS reward gradient scale")
    p.add_argument("--diffusion_norm", type=float, default=None,
                   help="SDE noise scale")
    p.add_argument("--init_n_particles", type=int, default=None,
                   help="RBF initial particles")
    p.add_argument("--search_steps", type=int, default=None,
                   help="Number of iterations for zero order search")
    p.add_argument("--search_radius", type=float, default=None,
                   help="Radius (lambda) for zero order search neighborhood")

    # Reward weights
    p.add_argument("--reward_w_clap", type=float, default=None,
                   help="CLAP reward weight")
    p.add_argument("--reward_w_aes", type=float, default=None,
                   help="AES reward weight")
    p.add_argument("--reward_w_beats", type=float, default=None,
                   help="BEATs reward weight")
    p.add_argument("--reward_w_aqa", type=float, default=None,
                   help="AQA (Qwen Audio) reward weight")

    # Execution control
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of test samples (-1 = all)")
    p.add_argument("--save_all_scores", action="store_true", default=None)
    p.add_argument("--log_interval", type=int, default=None)

    return p


def run(cfg):
    """Main execution function."""
    # --- Setup ---
    device = cfg.get("device", "cuda:0")
    # Put reward models on a separate GPU to avoid OOM
    reward_device = cfg.get("reward_device", "cuda:1" if torch.cuda.device_count() > 1 else "cpu")
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = cfg.get("output_dir", "./outputs/inference_scaling")
    scaling_method = cfg.get("scaling_method", "bon")
    os.makedirs(output_dir, exist_ok=True)

    # Save config for reproducibility
    config_save_path = os.path.join(output_dir, "config_used.json")
    with open(config_save_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # --- Print experiment info ---
    print("=" * 60)
    print("  Inference-Time Scaling for TangoFlux")
    print("=" * 60)
    print(f"  Method:       {scaling_method}")
    print(f"  Interpolant:  {cfg.get('interpolant', 'linear')}")
    print(f"  Sample mode:  {cfg.get('sample_method', 'ode')}")
    print(f"  Steps:        {cfg.get('num_inference_steps', 50)}")
    print(f"  Particles:    {cfg.get('n_particles', 8)}")
    print(f"  Duration:     {cfg.get('duration', 10)}s")
    print(f"  Model device: {device}")
    print(f"  Reward device:{reward_device}")
    print(f"  Search steps: {cfg.get('search_steps', 4)}")
    print(f"  Search radius: {cfg.get('search_radius', 0.80)}")
    print(f"  Randomize pivot: {cfg.get('randomize_pivot', False)}")
    print(f"  Output:       {output_dir}")
    print("=" * 60)

    # --- Load model ---
    print("\n[1/4] Loading TangoFlux model...")
    model_name = cfg.get("model_name", "declare-lab/TangoFlux")
    tango = TangoFluxInference(name=model_name, device=device)
    print(f"  Model loaded on {device}")

    # --- Load reward model (on separate device to avoid OOM) ---
    verifier_list = cfg.get("verifier_list", ["clap", "aes", "beats", "aqa"])
    print(f"\n[2/4] Loading reward ensemble on {reward_device}...")
    reward_weights = cfg.get("reward_weights", {"clap": 0.4, "aes": 0.1, "beats": 0.1, "aqa": 0.4})
    reward_model = EnsembleAudioReward(
        device=reward_device,
        verifier_list=verifier_list,
        clap_weight=reward_weights.get("clap", 0.4),
        aes_weight=reward_weights.get("aes", 0.1),
        beats_weight=reward_weights.get("beats", 0.1),
        aqa_weight=reward_weights.get("aqa", 0.4),
    )

    # --- Load dataset ---
    print("\n[3/4] Loading test dataset...")
    test_set_path = cfg.get("test_set_path")
    dataset = load_dataset("json", data_files={"test": test_set_path}, split="test")
    max_samples = cfg.get("max_samples", -1)
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"  Loaded {len(dataset)} samples from {test_set_path}")

    # --- Get scaling function ---
    scaling_fn = SCALING_METHODS[scaling_method]

    # --- Inference loop ---
    print(f"\n[4/4] Running inference with '{scaling_method}' method...")
    results = []
    duration = cfg.get("duration", 10)
    sample_rate = cfg.get("sample_rate", 44100)
    log_interval = cfg.get("log_interval", 10)

    audio_output_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_output_dir, exist_ok=True)

    for i, item in enumerate(tqdm(dataset, desc="Generating")):
        prompt = item.get("captions")
        original_name = os.path.basename(item.get("location", f"sample_{i:05d}.wav"))

        # Create a fresh prior for each prompt
        prior = TangoFluxPrior(
            model=tango.model,
            vae=tango.vae,
            text_prompt=prompt,
            duration=duration,
            guidance_scale=cfg.get("guidance_scale", 4.5),
            num_inference_steps=cfg.get("num_inference_steps", 50),
            interpolant=cfg.get("interpolant", "linear"),
            sample_method=cfg.get("sample_method", "ode"),
            diffusion_norm=cfg.get("diffusion_norm", 0.5),
            device=device,
        )

        # Build method-specific kwargs
        method_kwargs = {
            "prior": prior,
            "reward_model": reward_model,
            "prompt": prompt,
            "duration": duration,
            "sample_rate": sample_rate,
            "seed": seed + i,
        }

        if scaling_method == "bon":
            method_kwargs["n_samples"] = cfg.get("n_particles", 8)

        elif scaling_method == "particle_filter":
            method_kwargs["n_particles"] = cfg.get("n_particles", 4)
            method_kwargs["temperature"] = cfg.get("temperature", 1.0)
            method_kwargs["eval_interval"] = cfg.get("eval_interval", 5)

        elif scaling_method == "rbf":
            method_kwargs["init_n_particles"] = cfg.get("init_n_particles", 25)
            method_kwargs["max_nfe"] = cfg.get("max_nfe", 500)
            method_kwargs["eval_interval"] = cfg.get("eval_interval", 5)

        elif scaling_method == "dps":
            method_kwargs["guidance_strength"] = cfg.get("guidance_strength", 1.0)
            method_kwargs["eval_interval"] = cfg.get("eval_interval", 5)

        elif scaling_method == "zero_order":
            method_kwargs["n_candidates"] = cfg.get("n_particles", 4)
            method_kwargs["search_steps"] = cfg.get("search_steps", 5)
            method_kwargs["search_radius"] = cfg.get("search_radius", 5.0)
            method_kwargs["randomize_pivot"] = cfg.get("randomize_pivot", True)

        # Run scaling method
        try:
            waveform, best_score, extra = scaling_fn(**method_kwargs)

            # Save audio
            save_path = os.path.join(audio_output_dir, original_name)
            # Ensure mono or stereo format for saving
            audio_to_save = waveform[0].cpu()  # (channels, samples)
            torchaudio.save(save_path, audio_to_save, sample_rate)

            # Collect detailed scores
            sample_result = {
                "index": i,
                "prompt": prompt,
                "filename": original_name,
                "best_score": best_score,
            }

            # Save individual particle/sample scores
            if isinstance(extra, torch.Tensor):
                sample_result["particle_scores"] = extra.tolist()
            elif isinstance(extra, dict) and "rewards" in extra:
                sample_result["particle_scores"] = extra["rewards"]

            # Get detailed scores if save_all_scores is enabled
            if cfg.get("save_all_scores", True):
                with torch.no_grad():
                    details = reward_model.score_with_details(waveform, prompt)
                for key in ["clap", "aes", "beats", "aqa", "combined"]:
                    if key in details:
                        sample_result[f"{key}_score"] = details[key].mean().item()

            results.append(sample_result)

            if (i + 1) % log_interval == 0 or i == 0:
                tqdm.write(
                    f"  [{i+1}/{len(dataset)}] \"{prompt[:50]}...\" "
                    f"→ combined={best_score:.4f}"
                )

        except Exception as e:
            tqdm.write(f"  [Error] Sample {i}: {e}")
            tqdm.write(traceback.format_exc())
            results.append({
                "index": i,
                "prompt": prompt,
                "filename": original_name,
                "error": str(e),
            })
        finally:
            # Free GPU memory between samples
            del prior
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Save results ---
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Summary statistics ---
    successful = [r for r in results if "error" not in r]
    if successful:
        scores = [r["best_score"] for r in successful]
        print("\n" + "=" * 60)
        print("  Results Summary")
        print("=" * 60)
        print(f"  Successful: {len(successful)}/{len(results)}")
        print(f"  Mean score:  {np.mean(scores):.4f}")
        print(f"  Std score:   {np.std(scores):.4f}")
        print(f"  Min score:   {np.min(scores):.4f}")
        print(f"  Max score:   {np.max(scores):.4f}")

        if cfg.get("save_all_scores", True):
            clap_scores = [r.get("clap_score", 0) for r in successful]
            aes_scores = [r.get("aes_score", 0) for r in successful]
            beats_scores = [r.get("beats_score", 0) for r in successful]
            aqa_scores = [r.get("aqa_score", 0) for r in successful]
            print(f"\n  CLAP  mean: {np.mean(clap_scores):.4f}")
            print(f"  AES   mean: {np.mean(aes_scores):.4f}")
            print(f"  BEATs mean: {np.mean(beats_scores):.4f}")
            print(f"  AQA   mean: {np.mean(aqa_scores):.4f}")

        print(f"\n  Audio saved to: {audio_output_dir}")
        print(f"  Results saved to: {results_path}")
        print(f"  Config saved to: {config_save_path}")
    else:
        print("\n  [WARNING] No successful generations!")

    print("=" * 60)
    return results


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Load config file
    if os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        print(f"[WARNING] Config file '{args.config}' not found, using defaults.")
        cfg = {}

    # Merge CLI overrides
    cfg = merge_cli_overrides(cfg, args)

    # Validate
    method = cfg.get("scaling_method", "bon")
    if method not in SCALING_METHODS:
        print(f"[ERROR] Unknown scaling method: {method}")
        print(f"  Available: {list(SCALING_METHODS.keys())}")
        sys.exit(1)

    if method in ("particle_filter", "rbf") and cfg.get("sample_method") == "ode":
        print(f"[WARNING] '{method}' works best with SDE sampling. "
              f"Consider --sample_method sde for particle diversity.")

    # Run
    run(cfg)
