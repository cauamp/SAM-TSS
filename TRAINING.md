# RTMVSS Training Setup

## Prerequisites

1. **SAM2 Checkpoint**: Download SAM2 checkpoint file
   - Available from: https://github.com/facebookresearch/segment-anything-2
   - Supported models: `sam2.1_hiera_t`, `sam2.1_hiera_s`, `sam2.1_hiera_b+`, `sam2.1_hiera_l`
   - Update the path in `train.sh`: `--sam2-ckpt /path/to/sam2_hiera_large.pt`

2. **Dataset**: Ensure MVSeg dataset is configured
   - Update dataset path in `src/sam_tss/datasets/mvss_dataset.py`
   - Default location: `MVSeg_ROOT = "/set_your_path/MVSeg_Dataset/"`

3. **Environment**: Activate Python environment with required packages
   - PyTorch, SAM2, OpenCV, etc.

## Configuration

### train.sh Parameters

**Model Arguments:**
- `--model rtmvss_1.py` - Model file to use
- `--sam2-config sam2.1_hiera_l.yaml` - SAM2 architecture config
- `--sam2-ckpt /path/to/checkpoint.pt` - **REQUIRED**: Path to SAM2 checkpoint
- `--num-classes 26` - Number of classes (26 for MVSeg)
- `--num-frame-queries 5` - Number of frame-level queries
- `--num-video-queries 5` - Number of video-level queries for memory

**Training Arguments:**
- `--training` - Enable training mode
- `--baseline-mode` - Stage 1: Train without memory (warm-up)
- `--gpus 1` - Number of GPUs to use
- `--batch-size 2` - Batch size per GPU
- `--num-epochs 150` - Number of training epochs
- `--lr-start 2e-4` - Initial learning rate
- `--lr-strategy plateau_08` - Learning rate scheduler

**Video Segmentation:**
- `--stm-queue-size 3` - Memory queue size (number of reference frames)
- `--sample-rate 3` - Frame sampling rate
- `--win-size 4` - Window size (auto-calculated if -1)

## Training Stages

### Stage 1: Baseline Training (Warm-up)
Train model without memory mechanism to initialize weights:
```bash
./train.sh
```
or submit SLURM job:
```bash
sbatch job.sh
```

### Stage 2: Training with Memory
After Stage 1 completes, uncomment Stage 2 in `train.sh` and run again:
- Loads weights from Stage 1: `--weights save/training_base_rtmvss/model_base.pth`
- Removes `--baseline-mode` to enable memory
- Can use multiple GPUs: `--gpus 2`



