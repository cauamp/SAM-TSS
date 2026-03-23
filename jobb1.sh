#!/bin/bash
#SBATCH --time=58:00:00          # Time limit (HH:MM:SS) - 48 hours for full training
#SBATCH --output=run_%j/result.out   # Output file name 
#SBATCH --error=run_%j/log.err
#SBATCH --job-name=mvss-train # Name of the job

#SBATCH --account=def-vislearn

#SBATCH --mem=128G               # Request 128GB of memory
#SBATCH --cpus-per-task=16       # Request 16 CPU cores
#SBATCH --gpus-per-node=4      

export MODEL="rtmvss_3.py"
export GPUS=4
export SAVEDIR="exp3/run_$SLURM_JOB_ID"

# Training parameters
export BATCH_SIZE=4 
export ACCUMULATION_STEPS=2
export BASELINE_MODE=1
export LR_START=2e-4
export LR_STRATEGY="plateau_08"
export LOAD="./src/sam_tss/weights/dvisal.pt"

module --force purge # Clear all loaded modules

# Prevent PyTorch from trying to load Level Zero (Intel GPU backend)
export TORCH_USE_RTLD_GLOBAL=1
export PYTORCH_IGNORE_LEVEL_ZERO=1
export CLASS_QUERY_SIZE=1024
export RESIZE_MODE="og"

# Load required modules
module load StdEnv/2023 gcc/12.3
module load opencv/4.13.0

# Activate your virtual environment
# Option 1: If using a custom activation script
. ./activate.sh


# Create output directory for this job
mkdir -p $SAVEDIR
echo  "">$SAVEDIR/config.txt

echo "MODEL=${MODEL}" >> $SAVEDIR/config.txt
echo "GPUS=${GPUS}" >> $SAVEDIR/config.txt
echo "BATCH_SIZE=${BATCH_SIZE}" >> $SAVEDIR/config.txt
echo "ACCUMULATION_STEPS=${ACCUMULATION_STEPS}" >> $SAVEDIR/config.txt
echo "BASELINE_MODE=${BASELINE_MODE}" >> $SAVEDIR/config.txt
echo "LR_START=${LR_START}" >> $SAVEDIR/config.txt
echo "LR_STRATEGY=${LR_STRATEGY}" >> $SAVEDIR/config.txt
echo "LOAD=${LOAD}" >> $SAVEDIR/config.txt
echo "CLASS_QUERY_SIZE=${CLASS_QUERY_SIZE}" >> $SAVEDIR/config.txt
echo "RESIZE_MODE=${RESIZE_MODE}" >> $SAVEDIR/config.txt

# Set CUDA_VISIBLE_DEVICES based on allocated GPUs
#export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Run the training script
echo "Starting RTMVSS training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
#echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# Make train.sh executable
chmod +x ./train.sh

# Run training
./train.sh

echo "Training completed at $(date)"
