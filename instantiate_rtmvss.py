import torch
import argparse
from sam_tss.models.rtmvss import rtmvss
# from sam_tss.models.rdvsod import rdvsod as rtmvss

def create_args():
    """Create arguments object with required parameters"""
    args = argparse.Namespace()
    
    # SAM2 configuration
    args.sam2_config = "./src/sam_tss/models/sam2/configs/sam2.1/sam2.1_hiera_l_mvss.yaml"  # or sam2_hiera_l.yaml, sam2_hiera_s.yaml, sam2_hiera_t.yaml
    args.sam2_ckpt = "./src/sam_tss/models/sam2/sam2.1_hiera_large.pt"  # Update with actual checkpoint path
    
    
    # Query parameters
    args.num_frame_queries = 8
    args.num_video_queries = 8
    args.enable_memory = True
    args.enable_memory = True
    # Number of classes for segmentation
    args.num_classes = 2  # e.g., binary segmentation (background + foreground)
    
    return args


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create arguments
    args = create_args()
    
    # Instantiate the model
    model = rtmvss(args, device)
    model.to(device)
    model.eval()
    
    print("Model instantiated successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Example inference with dummy data
    batch_size = 1
    num_frames = 2
    height, width = 1024, 1024  # SAM2 expects 1024x1024 images
    
    # Create dummy RGB and depth images
    dummy_imgs = torch.randn(batch_size, num_frames, 3, height, width).to(device)
    dummy_depths = torch.randn(batch_size, num_frames, 1, height, width).to(device)
    
    print(f"\nRunning inference with dummy data:")
    print(f"Input RGB shape: {dummy_imgs.shape}")
    print(f"Input depth shape: {dummy_depths.shape}")
    
    # Forward pass (inference mode)
    with torch.no_grad():
        output = model(dummy_imgs, dummy_depths, is_mem=True, is_training=False, current_ti=0)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: [batch_size={batch_size}, num_classes={args.num_classes}, height={height}, width={width}]")


if __name__ == "__main__":
    main()
