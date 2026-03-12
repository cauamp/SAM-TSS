# Model Training

# Stage1: Training Warm-up for model
uv run   --model mvnet.py \
  --sam2-ckpt .ignore  \ 
--savedir ${SAVEDIR:-save/training_base_rtmvss} \
--backbone deeplab50 --training --baseline-mode --gpus $GPUS --lr-start 2e-4 --lr-strategy plateau_08 --num-epochs 150 --batch-size 2 --stm-queue-size 3 --sample-rate 3 --savedir save/training_base_deeplab

# Stage2: Training with memory
# CUDA_VISIBLE_DEVICES=0,1 python main.py --backbone deeplab50 --weights save/training_base_deeplab/model_base.pth --training --gpus 2 --lr-start 2e-4 --lr-strategy plateau_08 --num-epochs 200 --batch-size 2 --stm-queue-size 3 --sample-rate 3 --savedir save/training_msa_deeplab
