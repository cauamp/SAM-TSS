# Model Training

# Stage1: Training Warm-up for RTMVSS model
CUDA_LAUNCH_BLOCKING=1  uv run src/sam_tss/main.py \
  --model rtmvss_1.py \
  --sam2-config sam2.1_hiera_l.yaml \
  --sam2-ckpt ./src/sam_tss/models/sam2/sam2.1_hiera_large.pt \
  --load ./src/sam_tss/weights/dvisal.pt \
  --num-classes 26 \
  --num-frame-queries 30 \
  --num-video-queries 8 \
  --enable-memory \
  --training \
  --baseline-mode \
  --gpus 1 \
  --lr-start 2e-4 \
  --lr-strategy plateau_08 \
  --num-epochs 150 \
  --batch-size 1 \
  --accumulation-steps 4 \
  --stm-queue-size 3 \
  --sample-rate 3 \
  --savedir save/training_base_rtmvss

# Stage2: Training with memory (uncomment after Stage1 completes)
# CUDA_VISIBLE_DEVICES=0,1 python src/sam_tss/main.py \
#   --model rtmvss_1.py \
#   --sam2-config sam2.1_hiera_l.yaml \
#   --sam2-ckpt /path/to/sam2_hiera_large.pt \
#   --load save/training_base_rtmvss/model_base.pth \
#   --num-classes 26 \
#   --num-frame-queries 5 \
#   --num-video-queries 5 \
#   --enable-memory \
#   --training \
#   --gpus 2 \
#   --lr-start 2e-4 \
#   --lr-strategy plateau_08 \
#   --num-epochs 200 \
#   --batch-size 2 \
#   --accumulation-steps 2 \
#   --stm-queue-size 3 \
#   --sample-rate 3 \
#   --savedir save/training_msa_rtmvss
