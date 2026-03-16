# -*- coding: utf-8 -*-
import os
import time
import torch
from utils.utils import setup_seed
setup_seed(0)
from argparse import ArgumentParser

import torch.multiprocessing as mp
import torch.distributed as dist

import routines
from utils.saving import Saver
from utils.visualize import Dashboard, print_optimized_model_params
from utils.loading import load_model_from_file, load_checkpoint


def process(gpu, args):

    # Determine device type
    if torch.cuda.is_available() and args.device == 'cuda':
        device = f'cuda:{gpu}'
        device_type = 'cuda'
    elif torch.backends.mps.is_available() and args.device == 'mps':
        device = 'mps'
        device_type = 'mps'
        gpu = 0  # MPS doesn't use gpu index
    else:
        device = 'cpu'
        device_type = 'cpu'
        gpu = 0
    
    args.device_type = device_type
    args.device = device
    
    ############################################################
    # Only use distributed training for multi-GPU CUDA setups
    if args.world_size > 1 and device_type == 'cuda':
        rank = args.nr * args.gpus + gpu
        # Set the current CUDA device BEFORE initializing process group
        torch.cuda.set_device(gpu)
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=rank,
            device_id=torch.device(device)
        )
        args.distributed = True
    else:
        rank = 0
        args.distributed = False
    ############################################################

    torch.manual_seed(0)

    board = None
    if gpu == 0:
        if args.visualize and args.steps_plot > 0:
            board = Dashboard(args)

    # Save arguments for reference
    saver = Saver(args)
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(saver.save_dir, args.shallow_dec)

    model_path = os.path.join("src/sam_tss/models", args.model)     # Loading Model file, i.e., mvnet.py
    model = load_model_from_file(args, model_path, board, device, checkpoint)

    if gpu == 0:
        # Load Model and save a copy of it for reference
        saver.save_model_copy(model_path)
        print_optimized_model_params(model)
        saver.save_txtmodel(model, args.shallow_dec)

    # Use that model for training and/or inference
    if args.training:
        routines.train(args, board, saver, model, device, rank, checkpoint)
    else:
        routines.eval(args, board, saver, model, device, rank)

def main(args):
    # For single GPU or MPS, run directly without multiprocessing spawn
    if args.gpus == 1 or args.device in ['mps', 'cpu']:
        process(0, args)
    else:
        mp.spawn(process, nprocs=args.gpus, args=(args, ))

    print("========== PROCESSING FINISHED ===========")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', default="MVSeg")

    # Model selection
    parser.add_argument('--model', default="rtmvss_1.py")
    #parser.add_argument('--weights', default="trained_models/erfnet_pretrained.pth")
    parser.add_argument('--weights', default=False)
    parser.add_argument('--backbone', type=str, default="sam_based")
    
    # RTMVSS / SAM2 specific options
    parser.add_argument('--sam2-config', type=str, default="sam2.1_hiera_l.yaml", 
                        help='SAM2 model config (sam2.1_hiera_t/s/b+/l)')
    parser.add_argument('--sam2-ckpt', type=str, required=True,
                        help='Path to SAM2 checkpoint')
    parser.add_argument('--load', type=str, default=None,
                        help='Path to pretrained RTMVSS checkpoint (e.g., ./weights/dvisal.pt)')
    parser.add_argument('--num-classes', type=int, default=26,
                        help='Number of segmentation classes (26 for MVSeg)')
    parser.add_argument('--num-frame-queries', type=int, default=5,
                        help='Number of frame-level queries')
    parser.add_argument('--enable-memory', action='store_true', default=True,
                        help='Enable video memory mechanism')
    parser.add_argument('--num-video-queries', type=int, default=5,
                        help='Number of video-level queries for memory')

    # Network structure
    parser.add_argument('--backbone-nobn', action='store_true')
    parser.add_argument('--encoder-eval-mode', action='store_true')
    parser.add_argument('--decoder-eval-mode', action='store_true')
    parser.add_argument('--train-decoder', action='store_true')
    parser.add_argument('--train-encoder', action='store_true')
    parser.add_argument('--train-erfnet-shallow-dec', action='store_true', default=False)
    parser.add_argument('--shallow-dec', action='store_true', default=False)
    parser.add_argument('--use-orig-res', action='store_true')
    parser.add_argument('--always-decode', action='store_true')

    # Training options
    parser.add_argument('--eval-mode', action='store_true')
    parser.add_argument('--split-mode', default='val')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Number of gradient accumulation steps (simulates larger batch size)')
    parser.add_argument('--steps-loss', type=int, default=100)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)
    parser.add_argument('--lr-strategy', default='pow_09')
    parser.add_argument('--lr-start', default='5e-5')
    parser.add_argument('--loss', default="")

    # Multi GPU training
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--mport', type=str, default="8888")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'],
                        help='Device to use: cuda, mps (Apple Silicon), or cpu')

    # Debug output / Visdom options
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save-images', action='store_true')

    # Evaluation options
    parser.add_argument('--iouTrain', action='store_true', default=False)
    parser.add_argument('--iouVal', action='store_true', default=True)
    parser.add_argument('--save-all-vals', action='store_true')
    parser.add_argument('--baseline-mode', action='store_true')
    parser.add_argument('--baseline-path', default="<TO DEFINE>")

    # MVNet related options
    parser.add_argument('--win-size', type=int, default=-1, help='if not given, mem size + 1')
    parser.add_argument('--model-struct', type=str, default="original")
    parser.add_argument('--align-weights', action='store_true', default=True)
    parser.add_argument('--local-correlation', action='store_true', default=False)
    parser.add_argument('--corr-size', type=int, default=21)
    parser.add_argument('--learnable-constant', action='store_true')
    parser.add_argument('--memorize-first', action='store_true')
    parser.add_argument('--fusion-strategy', type=str, default="sigmoid-do1")

    parser.add_argument('--memory-strategy', type=str, default="all", help='all or random')
    parser.add_argument('--stm-queue-size', type=int, default=3)
    parser.add_argument('--sample-rate', type=int, default=1)
    
    parser.add_argument('--class-query-size', type=int, default=1024, help='Dimension of the class query used in sparse embedding')
    parser.add_argument('--resize-mode', type=str, default="og", choices=['og', 'sam', 'sam1'], help='Resize mode for input images (original or SAM-specific)')

    # Augment related options
    parser.add_argument('--random-crop', action='store_true', default=True)

    args = parser.parse_args()
    
    # Validate RTMVSS-specific arguments
    if args.model == "rtmvss_1.py" or args.model == "rtmvss.py":
        if not hasattr(args, 'sam2_ckpt') or not args.sam2_ckpt:
            print("Error: --sam2-ckpt is required for RTMVSS model")
            exit(1)

    #########################################################
    args.world_size = args.gpus * args.nodes                #
    os.environ['MASTER_ADDR'] = 'localhost'                 #
    os.environ['MASTER_PORT'] = args.mport                  #
    #########################################################

    if args.eval_mode == args.training:
        print("Cannot be at the same time in training and evaluation mode!")
        exit(1)

    
    # Options simplifications
    if args.eval_mode:
        args.num_epochs = 1
        args.batch_size = 1

    if args.win_size == -1:
        args.win_size = args.stm_queue_size + 1

    start_time = time.time()
    
    main(args)
    
    print("Total time: {0:6.4f} s".format(time.time() - start_time))
