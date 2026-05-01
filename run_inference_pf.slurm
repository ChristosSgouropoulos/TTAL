#!/bin/bash
#SBATCH --job-name=tango_pf
#SBATCH --output=tango_pf_%j.log
#SBATCH --error=tango_pf_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
export HF_HOME=/leonardo/home/userexternal/csgourop/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# load minimal modules
module load cuda

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tango

# go to code folder
cd /leonardo/home/userexternal/csgourop/TTAL

# run
python main_inference_scaling.py --config inference_scaling_pf.json
