import torch
import os

from models.mvnet import MVNet
from models.rtmvss_1 import rtmvss
from models.rtmvss_2 import rtmvss as rtmvss2
from models.rtmvss_3 import rtmvss as rtmvss3
from models.rtmvss_3a import rtmvss as rtmvss3a
from models.rtmvss_3b import rtmvss as rtmvss3b
from models.rtmvss_3c import rtmvss as rtmvss3c
from models.rtmvss_3d import rtmvss as rtmvss3d
from models.rtmvss_3e import rtmvss as rtmvss3e
from models.rtmvss_40 import rtmvss as rtmvss4
from models.rtmvss_5 import rtmvss as rtmvss5
from models.rtmvss_6a import build_rtmvss6_from_args as build_rtmvss6_from_argsa

models = {
    "mvnet": MVNet,
    "rtmvss_1": rtmvss,
    "rtmvss_2": rtmvss2,
    "rtmvss_3": rtmvss3,
    "rtmvss_3a": rtmvss3a,
    "rtmvss_3c": rtmvss3c,
    "rtmvss_3d": rtmvss3d,
    "rtmvss_3b": rtmvss3b,
    "rtmvss_40": rtmvss4,
    "rtmvss_5": rtmvss5,
    "rtmvss_6a": build_rtmvss6_from_argsa,
}


def load_model_from_file(args, model_path, board, device, checkpoint):

    if not os.path.exists(model_path):
        print("Could not load model, file does not exist: ", model_path)
        exit(1)

    # Determine if this is rank 0 or single device
    print_all_logs = (not args.distributed or device == args.device or 
                      (isinstance(device, str) and ('cuda:0' in device or device in ['mps', 'cpu'])))

    model_name = str(os.path.basename(model_path).split('.')[0])
    
    # Handle different model constructors
    if 'rtmvss'in model_name :
        # rtmvss models take (args, device) as constructor arguments
        model = models[model_name](args, device=device)
    else:
        # MVNet and other models take (args, print_all_logs, board)
        model = models[model_name](args, print_all_logs=print_all_logs, board=board)
    
    if print_all_logs:
        print("Loaded model file: ", model_path)

    # Move model to device
    model = model.to(device)

    # Only use DistributedDataParallel for multi-GPU CUDA training
    if args.distributed and args.device_type == 'cuda':
        # Extract GPU id from device string for DDP
        gpu_id = int(device.split(':')[1]) if isinstance(device, str) and ':' in device else device
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        if args.gpus > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        # Wrap in a simple wrapper to maintain .module attribute for compatibility
        class SimpleWrapper(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)
        model = SimpleWrapper(model)

    # Handle weight loading
    if model_name in ["rtmvss_1", "rtmvss"]:
        # RTMVSS-specific loading: merge SAM2 checkpoint with model checkpoint
        if hasattr(args, 'load') and args.load:
            if os.path.exists(args.load):
                if print_all_logs:
                    print("Loading RTMVSS checkpoint: ", args.load)
                    print("Loading SAM2 checkpoint: ", args.sam2_ckpt)
                
                # Load checkpoints
                checkpoint_weights = torch.load(args.load, map_location=lambda storage, loc: storage)
                sam2_state = torch.load(args.sam2_ckpt, map_location=lambda storage, loc: storage)
                
                # Merge SAM2 weights with checkpoint
                new_state_dict = {}
                
                # Add SAM2 weights with "sam2." prefix
                if "model" in sam2_state:
                    for k, v in sam2_state["model"].items():
                        new_state_dict["module.sam2." + k] = v  # Add module prefix for DDP
                else:
                    for k, v in sam2_state.items():
                        new_state_dict["module.sam2." + k] = v
                
                # Add checkpoint weights with proper key handling
                # Check if checkpoint keys have "module." prefix
                sample_key = next(iter(checkpoint_weights.keys()))
                has_module_prefix = sample_key.startswith("module.")
                
                for k, v in checkpoint_weights.items():
                    if 'mixer2.4.' in k:
                        print("Skipping loading weights for key: ", k)
                        continue
                    if has_module_prefix:
                        # Keys already have module prefix, use as-is
                        new_state_dict[k] = v
                    else:
                        # Add module prefix for DDP compatibility
                        new_state_dict["module." + k] = v
                
                model.load_state_dict(new_state_dict, strict=False)
                
                if print_all_logs:
                    print("Loaded merged weights: SAM2 + RTMVSS checkpoint")
            else:
                print("Could not load checkpoint, file does not exist: ", args.load)
        elif args.sam2_ckpt:
            # Only load SAM2 weights if no checkpoint provided
            if os.path.exists(args.sam2_ckpt):
                if print_all_logs:
                    print("Loading SAM2 checkpoint only: ", args.sam2_ckpt)
                
                sam2_state = torch.load(args.sam2_ckpt, map_location=lambda storage, loc: storage)
                new_state_dict = {}
                
                # Add SAM2 weights with proper prefix
                if "model" in sam2_state:
                    for k, v in sam2_state["model"].items():
                        new_state_dict["module.sam2." + k] = v
                else:
                    for k, v in sam2_state.items():
                        new_state_dict["module.sam2." + k] = v
                
                model.load_state_dict(new_state_dict, strict=False)
                
                if print_all_logs:
                    print("Loaded SAM2 weights only")
    else:
        # Original loading logic for MVNet and other models
        weights_path = args.weights
        if weights_path:
            if os.path.exists(weights_path):
                if print_all_logs:
                    print("Loading weights file: ", weights_path)
            else:
                print("Could not load weights, file does not exist: ", weights_path)

            if checkpoint is not None:
                weights_dict = checkpoint
            else:
                weights_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)

            model.load_state_dict(weights_dict)

            if print_all_logs:
                print("Loaded weights:", weights_path)

    if args.training:
        model.train()
    else:
        model.eval()

    #filter_model_params_optimization(args, model)

    return model



def load_checkpoint(save_dir, enc):
    tag = "_enc" if enc else ""
    checkpoint_path = os.path.join(save_dir, "checkpoint{}.pth.tar".format(tag))

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint file: ", checkpoint_path)
    else:
        print("Could not load checkpoint, file does not exist: ", checkpoint_path)

    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
