import torch
import torch.nn.functional as F
import argparse
from sam_tss.models.rtmvss_7c import rtmvss
# from sam_tss.models.mvnet import MVNet as rtmvss


def create_args():
    """Create arguments object with required parameters"""
    args = argparse.Namespace()

    # SAM2 configuration (matching train.sh)
    args.sam2_config = "sam2.1_hiera_l_mvss.yaml"
    args.sam2_ckpt = "./src/sam_tss/models/sam2/sam2.1_hiera_large.pt"
    args.load = "./src/sam_tss/weights/dvisal.pt"

    # Query parameters (matching train.sh)
    args.num_frame_queries = 30
    args.num_video_queries = 5
    args.enable_memory = True
    args.dataset = "MVSeg"
    args.model_struct = "original"
    args.baseline_mode = False  # Set to True to test baseline (no memory, only last frame)
    # Number of classes for segmentation (matching train.sh - MVNet has 26 classes)
    args.memory_strategy = "all"
    args.stm_queue_size = 3
    args.sample_rate = 3
    args.backbone = "deeplab50"  # Use SAM-based backbone for testing
    args.num_classes = 26

    # Training parameters (matching train.sh)
    args.training = True
    args.win_size = 3  # stm_queue_size from train.sh
    args.always_decode = False  # baseline_mode typically decodes only last frame

    return args


def main():
    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Create arguments
    args = create_args()

    # Instantiate the model
    model = rtmvss(args, device)
    model.to(device)
    model.train()  # Set to training mode

    print("Model instantiated successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training parameters from train.sh
    batch_size = 2  # train.sh uses 2
    num_frames = 2  # train.sh uses 3 (stm_queue_size)
    # IMPORTANT: Use 1024x1024 to match SAM2's positional encoding configuration
    # SAM2 was configured for image_size=1024, producing 64x64 main features
    # Using other resolutions will cause dimension mismatch in the transformer
    height, width = (320, 480)

    # Create dummy RGB and thermal images
    dummy_imgs = torch.randn(batch_size, num_frames, 3, height, width, requires_grad=True).to(device)
    dummy_thermal = torch.randn(batch_size, num_frames, 3, height, width, requires_grad=True).to(device)

    # Create dummy ground truth labels
    dummy_labels = torch.randint(0, args.num_classes, (batch_size, 1, height, width)).to(device)  # [B, 1, H, W]

    print("\nInput shapes:")
    print(f"  RGB: {dummy_imgs.shape}")
    print(f"  Thermal: {dummy_thermal.shape}")
    print(f"  Labels: {dummy_labels.shape}")

    # Forward pass
    print(f"\n{'=' * 80}")
    print("FORWARD PASS")
    print(f"{'=' * 80}")

    main_pred, aux_rgb, aux_thermal, aux_fusion, features = model(dummy_imgs, dummy_thermal, step=0, epoch=0)

    print("\nForward outputs:")
    print(f"  Main predictions (logits): {main_pred.shape}")
    print(f"  Aux RGB: {aux_rgb.shape if aux_rgb is not None else None}")
    print(f"  Aux thermal: {aux_thermal.shape if aux_thermal is not None else None}")
    print(f"  Aux fusion (logits): {aux_fusion.shape if aux_fusion is not None else None}")
    print(f"  Features: {features.shape if features is not None else None}")

    exit()
    # Backward pass
    print(f"\n{'=' * 80}")
    print("BACKWARD PASS")
    print(f"{'=' * 80}")

    # Compute loss (simplified - actual training uses more complex loss)
    # Main prediction loss
    main_pred_for_loss = main_pred.squeeze(1)  # [B, 1, C, H, W] -> [B, C, H, W]
    loss_main = F.cross_entropy(main_pred_for_loss, dummy_labels.squeeze(1))

    print("\nLoss computation:")
    print(f"  Main loss (CrossEntropy): {loss_main.item():.4f}")

    # Auxiliary loss if available
    if aux_fusion is not None:
        aux_fusion_for_loss = aux_fusion.squeeze(1)  # [B, 1, C, H, W] -> [B, C, H, W]
        loss_aux = F.cross_entropy(aux_fusion_for_loss, dummy_labels.squeeze(1))
        print(f"  Aux fusion loss (CrossEntropy): {loss_aux.item():.4f}")

        # Total loss with auxiliary weight
        total_loss = loss_main + 0.4 * loss_aux  # MVNet typically uses 0.4 weight for aux
        print(f"  Total loss: {total_loss.item():.4f}")
    else:
        total_loss = loss_main

    # Backward pass
    print("\nComputing gradients...")
    total_loss.backward()

    # Check gradients
    grad_count = 0
    grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            grad_norm += param.grad.norm().item()

    print(f"  Gradients computed for {grad_count} parameters")
    print(f"  Total gradient norm: {grad_norm:.4f}")

    print(f"\n{'=' * 80}")
    print("SUCCESS!")
    print(f"{'=' * 80}")
    print("\n✓ Model instantiation: OK")
    print("✓ Forward pass: OK")
    print("✓ Backward pass: OK")
    print("✓ Gradients: OK")
    print("\nThe model is ready for training!")
    print("\nConfiguration (from train.sh):")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Temporal window: {num_frames}")
    print(f"  - Number of classes: {args.num_classes}")
    print(f"  - Frame queries: {args.num_frame_queries}")
    print(f"  - Video queries: {args.num_video_queries}")
    print(f"  - Enable memory: {args.enable_memory}")


if __name__ == "__main__":
    main()
