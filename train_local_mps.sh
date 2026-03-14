#!/bin/bash
# Local Training with MPS (Apple Silicon GPU)

# Stage1
# uv run src/sam_tss/main.py \
#   --model rtmvss_3.py \
#   --sam2-config sam2.1_hiera_l.yaml \
#   --sam2-ckpt ./src/sam_tss/models/sam2/sam2.1_hiera_large.pt \
#   --load ./src/sam_tss/weights/dvisal.pt \
#   --num-classes 26 \
#   --num-frame-queries 30 \
#   --num-video-queries 8 \
#   --enable-memory \
#   --training \
#   --baseline-mode \
#   --device mps \
#   --gpus 1 \
#   --lr-start 2e-4 \
#   --lr-strategy plateau_08 \
#   --num-epochs 150 \
#   --batch-size 1 \
#   --stm-queue-size 3 \
#   --sample-rate 3 \
#   --savedir .ignore/training_base_rtmvss_mps


#Stage2:
uv run src/sam_tss/main.py \
  --model rtmvss_3.py \
  --sam2-config sam2.1_hiera_l.yaml \
  --sam2-ckpt ./src/sam_tss/models/sam2/sam2.1_hiera_large.pt \
  --load runs/run_57653716/model_best.pth \
  --num-classes 26 \
  --num-frame-queries 30 \
  --num-video-queries 8 \
  --enable-memory \
  --training \
  --lr-start 2e-4 \
  --lr-strategy plateau_08 \
  --num-epochs 200 \
  --batch-size 2 \
  --accumulation-steps 16 \
  --stm-queue-size 3 \
  --sample-rate 3 \
  --savedir ${SAVEDIR:-save/training_msa_rtmvss} \
  --gpus 1



