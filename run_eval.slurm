#!/bin/bash
#SBATCH --job-name=tango_eval
#SBATCH --output=tango_eval_%j.log
#SBATCH --error=tango_eval_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Override these via --export when submitting, e.g.:
#   sbatch --export=ALL,GEN_DIR=/path/to/run/audio,RUN_TAG=pso run_eval.slurm
GEN_DIR=${GEN_DIR:-/leonardo_scratch/fast/EUHPC_D34_102/csgourop/inference_scaling_experiments/inference_scaling_zero_order_ode_linear_6_candidates_5_search_radius_099_CLAP_YES_randomize_pivot_new_prior/audio}
RUN_TAG=${RUN_TAG:-zero_order}

# Weight caches on scratch (compute nodes have no internet)
export CLAP_CACHE_DIR=/leonardo_scratch/fast/EUHPC_D34_102/csgourop/model_weights/clap_score
export TORCH_HOME=/leonardo_scratch/fast/EUHPC_D34_102/csgourop/model_weights/torch_home
export HF_HOME=/leonardo/home/userexternal/csgourop/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Keep per-run tmp preprocessed dir distinct so parallel jobs don't clobber each other
FIXED_DIR=/leonardo_scratch/fast/EUHPC_D34_102/csgourop/eval_tmp/fixed_${RUN_TAG}_${SLURM_JOB_ID}

module load cuda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /leonardo_scratch/fast/EUHPC_D34_102/csgourop/envs/eval

cd /leonardo/home/userexternal/csgourop/TTAL/stable-audio-metrics

python eval_updated.py \
    --config eval_config.json \
    --raw_generated_path "$GEN_DIR" \
    --fixed_generated_path "$FIXED_DIR"
