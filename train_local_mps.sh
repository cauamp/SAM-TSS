#!/bin/bash
# Local Training with MPS (Apple Silicon GPU)

# Stage1: Training Warm-up for RTMVSS model on MPS
uv run src/sam_tss/main.py \
  --model rtmvss_2.py \
  --sam2-config sam2.1_hiera_l.yaml \
  --sam2-ckpt ./src/sam_tss/models/sam2/sam2.1_hiera_large.pt \
  --load ./src/sam_tss/weights/dvisal.pt \
  --num-classes 26 \
  --num-frame-queries 30 \
  --num-video-queries 8 \
  --enable-memory \
  --training \
  --baseline-mode \
  --device mps \
  --gpus 1 \
  --lr-start 2e-4 \
  --lr-strategy plateau_08 \
  --num-epochs 150 \
  --batch-size 1 \
  --stm-queue-size 3 \
  --sample-rate 3 \
  --savedir .ignore/training_base_rtmvss_mps

# For testing on CPU instead, change --device mps to --device cpu
# For smaller test runs, reduce --num-epochs to like 5
