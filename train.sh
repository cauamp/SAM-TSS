# Model Training

# Enable NCCL debugging and extend timeout for slower operations
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export TORCH_NCCL_TRACE_BUFFER_SIZE=10000
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# Uncomment below for production runs (disables detailed debugging)
# CUDA_LAUNCH_BLOCKING=1

# Stage1
uv run src/sam_tss/main.py \
  --model ${MODEL:-.py} \
  --sam2-config sam2.1_hiera_l.yaml \
  --sam2-ckpt ./src/sam_tss/models/sam2/sam2.1_hiera_large.pt \
  $([ -n "${LOAD}" ] && echo "--load ${LOAD}") \
  --num-classes 26 \
  --num-frame-queries 30 \
  --num-video-queries 8 \
  --enable-memory \
  --training \
  $([ "${BASELINE_MODE:-0}" = "1" ] && echo "--baseline-mode") \
  --gpus ${GPUS:-4}   \
  --lr-start ${LR_START:-2e-4} \
  --lr-strategy ${LR_STRATEGY:-plateau_08} \
  --num-epochs 150 \
  --batch-size ${BATCH_SIZE:-2} \
  --accumulation-steps ${ACCUMULATION_STEPS:-8} \
  --stm-queue-size 3 \
  --sample-rate 3 \
  --class-query-size ${CLASS_QUERY_SIZE:-1024} \
  --resize-mode ${RESIZE_MODE:-og} \
  --savedir ${SAVEDIR:-save/training_base_rtmvss}

# Stage2
# CUDA_VISIBLE_DEVICES=0,1 python src/sam_tss/main.py \
#   --model ${MODEL:-.py} \
#   --sam2-config sam2.1_hiera_l.yaml \
#   --sam2-ckpt ./src/sam_tss/models/sam2/sam2.1_hiera_large.pt \
#   --load run_57653716/model_best.pth \
#   --num-classes 26 \
#   --num-frame-queries 30 \
#   --num-video-queries 8 \
#   --enable-memory \
#   --training \
#   --gpus $GPUS \
#   --lr-start 2e-4 \
#   --lr-strategy plateau_08 \
#   --num-epochs 200 \
#   --batch-size 2 \
#   --accumulation-steps 16 \
#   --stm-queue-size 3 \
#   --sample-rate 3 \
#   --savedir ${SAVEDIR:-save/training_msa_rtmvss}
