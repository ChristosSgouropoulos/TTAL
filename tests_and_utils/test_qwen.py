#!/usr/bin/env python3
"""
Standalone test for the Qwen Audio verifier (AQAScore).

Usage:
    python test_qwen.py                          # uses dummy noise
    python test_qwen.py --audio path/to/file.wav # uses a real file
"""

import argparse
import torch
import torchaudio

from verifiers import QwenAudioWrapper, AQAScoreRewardModel


def main():
    parser = argparse.ArgumentParser(description="Test AQAScore (Qwen Audio) verifier")
    parser.add_argument("--audio", type=str, default=None, help="Path to a .wav file")
    parser.add_argument("--prompt", type=str, default="A person talking over heavy wind.",
                        help="Text prompt to score against")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load or generate audio ---
    if args.audio:
        waveform, sr = torchaudio.load(args.audio)
        print(f"Loaded {args.audio}  shape={waveform.shape}  sr={sr}")
        # Add batch dimension: (C, T) -> (1, C, T)
        waveforms = waveform.unsqueeze(0)
    else:
        sr = 44100
        print(f"No audio file provided — using 2 s of random noise at {sr} Hz")
        waveforms = torch.randn(1, 1, sr * 2)  # (N=1, C=1, T)

    # --- Create reward model (self-contained — loads Qwen internally) ---
    print("Loading AQAScoreRewardModel (Qwen2-Audio-7B-Instruct) ...")
    reward = AQAScoreRewardModel(device=device)

    # --- Score ---
    print(f"\nPrompt : '{args.prompt}'")
    scores = reward(waveforms, args.prompt)

    print(f"\nAQAScore  P(yes) = {scores[0].item():.4f}")

    # Also show raw logits for transparency
    question = reward._build_question(args.prompt)
    raw = reward.audio_llm(waveforms[0], question)
    print(f"Raw logits — Yes: {raw['logits_yes']:.4f}  No: {raw['logits_no']:.4f}")


if __name__ == "__main__":
    main()
