#!/bin/bash
#SBATCH --time=48:00:00          # Time limit (HH:MM:SS) - 48 hours for full training
#SBATCH --output=run_%j/result.out   # Output file name 
#SBATCH --error=run_%j/log.err
#SBATCH --job-name=rtmvss-train  # Name of the job

#SBATCH --account=def-vislearn

#SBATCH --mem=128G               # Request 128GB of memory
#SBATCH --cpus-per-task=16       # Request 16 CPU cores
#SBATCH --gpus-per-node=1        # Request 1 GPU per node (adjust for multi-GPU)

module --force purge # Clear all loaded modules

# Prevent PyTorch from trying to load Level Zero (Intel GPU backend)
export TORCH_USE_RTLD_GLOBAL=1
export PYTORCH_IGNORE_LEVEL_ZERO=1

# Load required modules
module load StdEnv/2023 gcc/12.3
module load opencv/4.13.0

# Activate your virtual environment
# Option 1: If using a custom activation script
. ./activate.sh

# Option 2: If using conda (uncomment if needed)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate torch222

# Create output directory for this job
mkdir -p run_$SLURM_JOB_ID

# Set CUDA_VISIBLE_DEVICES based on allocated GPUs
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Run the training script
echo "Starting RTMVSS training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# Make train.sh executable
chmod +x ./train.sh

# Run training
./train.sh

echo "Training completed at $(date)"
