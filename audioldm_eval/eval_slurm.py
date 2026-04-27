"""Slurm-friendly entry point for audioldm_eval.

Wraps EvaluationHelperParallel with CLI args so the same script can evaluate
multiple generation runs on Leonardo without editing source.
"""
import argparse
import torch
from audioldm_eval import EvaluationHelperParallel


REFERENCE_PATH = "/leonardo_scratch/fast/EUHPC_D34_102/csgourop/reference_886_stereo"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True, help="Directory of generated audio (.wav files)")
    p.add_argument("--ref", default=REFERENCE_PATH, help="Reference audio directory")
    p.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    p.add_argument("--backbone", default="cnn14", choices=["cnn14", "mert"])
    p.add_argument("--sample-rate", type=int, default=16000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluator = EvaluationHelperParallel(args.sample_rate, args.gpus, backbone=args.backbone)
    metrics = evaluator.main(
        args.gen,
        args.ref,
        limit_num=None,
    )
    print(metrics)
