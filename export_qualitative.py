import ast
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
SAM_TSS_ROOT = SRC_ROOT / "sam_tss"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SAM_TSS_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_TSS_ROOT))

from sam_tss.datasets.helpers import DATASETS_DICT
from sam_tss.datasets.mvss_dataset import cmap
from sam_tss.utils.loading import load_model_from_file
from sam_tss.utils.utils import class_to_RGB


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_namespace_file(args_file: Path) -> Namespace:
    content = args_file.read_text().strip()
    expr = ast.parse(content, mode="eval").body
    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != "Namespace":
        raise ValueError(f"Unsupported args.txt format in {args_file}")

    values = {}
    for kw in expr.keywords:
        if kw.arg is None:
            continue
        values[kw.arg] = ast.literal_eval(kw.value)

    defaults = {
        "dataset": "MVSeg",
        "model": "rtmvss_3.py",
        "weights": False,
        "backbone": "sam_based",
        "num_classes": 26,
        "num_frame_queries": 30,
        "enable_memory": True,
        "num_video_queries": 8,
        "backbone_nobn": False,
        "encoder_eval_mode": False,
        "decoder_eval_mode": False,
        "train_decoder": False,
        "train_encoder": False,
        "train_erfnet_shallow_dec": False,
        "shallow_dec": False,
        "use_orig_res": False,
        "always_decode": False,
        "eval_mode": True,
        "split_mode": "test",
        "training": False,
        "resume": False,
        "num_epochs": 1,
        "batch_size": 1,
        "accumulation_steps": 1,
        "steps_loss": 100,
        "steps_plot": 50,
        "epochs_save": 0,
        "lr_strategy": "plateau_08",
        "lr_start": "2e-4",
        "loss": "",
        "nodes": 1,
        "gpus": 1,
        "nr": 0,
        "mport": "8888",
        "device": "cpu",
        "savedir": "./tmp",
        "port": 8097,
        "num_workers": 0,
        "visualize": False,
        "save_images": False,
        "iouTrain": False,
        "iouVal": False,
        "save_all_vals": False,
        "baseline_mode": True,
        "baseline_path": "<TO DEFINE>",
        "win_size": 4,
        "model_struct": "original",
        "align_weights": True,
        "local_correlation": False,
        "corr_size": 21,
        "learnable_constant": False,
        "memorize_first": False,
        "fusion_strategy": "sigmoid-do1",
        "memory_strategy": "all",
        "stm_queue_size": 3,
        "sample_rate": 3,
        "class_query_size": 1024,
        "resize_mode": "og",
        "random_crop": False,
        "sam2_config": "sam2.1_hiera_l.yaml",
        "sam2_ckpt": "./src/sam_tss/models/sam2/sam2.1_hiera_large.pt",
        "load": None,
    }

    defaults.update(values)
    args = Namespace(**defaults)

    if getattr(args, "win_size", -1) == -1:
        args.win_size = args.stm_queue_size + 1

    return args


def pick_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def resolve_checkpoint(run_dir: Path, ckpt_arg: Optional[str]) -> Path:
    if ckpt_arg:
        ckpt_path = Path(ckpt_arg)
        return ckpt_path if ckpt_path.is_absolute() else (PROJECT_ROOT / ckpt_path)

    preferred = [
        run_dir / "model_best.pth",
        run_dir / "model_best.pth.tar",
        run_dir / "checkpoint.pth.tar",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    all_candidates = sorted(list(run_dir.glob("*.pth")) + list(run_dir.glob("*.pth.tar")))
    if all_candidates:
        return all_candidates[0]

    raise FileNotFoundError(f"No .pth/.pth.tar checkpoint found in {run_dir}")


def denormalize_to_uint8(image_tensor: torch.Tensor, backbone: str) -> np.ndarray:
    image = image_tensor.detach().cpu().clone()
    if backbone == "sam_based":
        image = image * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1)
    image_np = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return image_np


def overlay(rgb: np.ndarray, seg_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mixed = (1.0 - alpha) * rgb.astype(np.float32) + alpha * seg_rgb.astype(np.float32)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def save_panel(
    out_dir: Path,
    split: str,
    index: int,
    file_path: str,
    rgb: np.ndarray,
    thermal: np.ndarray,
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    aux_pred_map: Optional[np.ndarray] = None,
) -> None:
    color_map = cmap()
    gt_rgb = class_to_RGB(gt_map, N=len(color_map), cmap=color_map)
    pred_rgb = class_to_RGB(pred_map, N=len(color_map), cmap=color_map)

    panel = np.concatenate(
        [
            rgb,
            thermal,
            gt_rgb,
            pred_rgb,
            overlay(rgb, pred_rgb),
        ],
        axis=1,
    )
    if aux_pred_map is not None:
        print("Adding auxiliary prediction to panel")
        aux_pred_rgb = class_to_RGB(aux_pred_map, N=len(color_map), cmap=color_map)
        panel = np.concatenate([panel, aux_pred_rgb, overlay(rgb, aux_pred_rgb)], axis=1)

    src = Path(file_path)
    frame = src.stem
    video = src.parent.parent.name if len(src.parts) >= 3 else "sample"

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    panel_path = split_dir / f"{index:03d}_{video}_{frame}_panel.png"
    pred_path = split_dir / f"{index:03d}_{video}_{frame}_pred.png"

    Image.fromarray(panel).save(panel_path)
    Image.fromarray(pred_rgb).save(pred_path)

    # rename out_dir/overlay and out_dir/pred_rgb to out_dir/{split}_overlay and out_dir/{split}_pred_rgb
    new_overlay_path = split_dir / f"{index:03d}_{video}_{frame}_lowres_overlay.png"
    new_pred_path = split_dir / f"{index:03d}_{video}_{frame}_lowres_pred.png"
    new_img_path = split_dir / f"{index:03d}_{video}_{frame}_lowres_img.png"

    # use mv to rename the files
    # os.rename("./tmp/overlay.png", new_overlay_path)
    # os.rename("./tmp/pred_rgb.png", new_pred_path)
    # os.rename("./tmp/lowres_img.png", new_img_path)


def run_split(
    args: Namespace,
    device: str,
    model: torch.nn.Module,
    split: str,
    n_samples: int,
    output_dir: Path,
    num_workers: int,
) -> int:
    if n_samples <= 0:
        return 0

    interval = [args.win_size - 1, 0]
    dataset = DATASETS_DICT[args.dataset](
        args,
        split,
        co_transform=False,
        shallow_dec=args.shallow_dec,
        augment=False,
        interval=interval,
        print_all_logs=False,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    exported = 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if exported >= n_samples:
                break

            images, thermals, labels, _, _, _, file_path, _, _, _ = batch

            images = images.to(device)
            thermals = thermals.to(device)
            labels = labels.to(device)

            model.module.reset_hidden_state()
            outputs = model(images, thermals, step=step, epoch=0)
            probabilities = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            aux_probs = outputs[-2] if isinstance(outputs, (list, tuple)) and len(outputs) > 2 else None
            aux_pred_labels = aux_probs.max(dim=2, keepdim=True)[1] if aux_probs is not None else None
            pred_labels = probabilities.max(dim=2, keepdim=True)[1]

            t = images.size(1) - 1 if args.always_decode else 0
            pred_map = pred_labels[0, t, 0].cpu().numpy().astype(np.uint8)
            aux_pred_map = (
                aux_pred_labels[0, t, 0].cpu().numpy().astype(np.uint8) if aux_pred_labels is not None else None
            )
            gt_map = labels[0, 0, 0].cpu().numpy().astype(np.uint8)
            rgb = denormalize_to_uint8(images[0, -1], args.backbone)
            thermal = denormalize_to_uint8(thermals[0, -1], args.backbone)

            save_panel(output_dir, split, exported, file_path[0], rgb, thermal, gt_map, pred_map, aux_pred_map)
            exported += 1

    return exported


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Folder containing args.txt and model checkpoints")
    parser.add_argument("--args-file", default=None, help="Optional path to args.txt (defaults to <run-dir>/args.txt)")
    parser.add_argument("--ckpt", default=None, help="Optional path to checkpoint (.pth or .pth.tar)")
    parser.add_argument("--output-dir", default="./tmp", help="Output folder for qualitative results")
    parser.add_argument("--num-train", type=int, default=10, help="Number of train samples to export")
    parser.add_argument("--num-test", type=int, default=10, help="Number of test samples to export")
    parser.add_argument("--train-split", default="train", help="Train split filename stem (e.g. train for train.txt)")
    parser.add_argument("--test-split", default="test", help="Test split filename stem (e.g. test for test.txt)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    cli = parser.parse_args()

    run_dir = Path(cli.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    args_file = Path(cli.args_file) if cli.args_file else (run_dir / "args.txt")
    if not args_file.is_absolute():
        args_file = PROJECT_ROOT / args_file
    if not args_file.exists():
        raise FileNotFoundError(f"args.txt not found: {args_file}")

    args = parse_namespace_file(args_file)

    checkpoint_path = resolve_checkpoint(run_dir, cli.ckpt)
    output_dir = Path(cli.output_dir) / run_dir.name
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(cli.device)

    args.training = False
    args.eval_mode = True
    args.distributed = False
    args.world_size = 1
    args.gpus = 1
    args.nr = 0
    args.device_type = device
    args.device = device
    args.batch_size = 1
    args.num_workers = cli.num_workers
    args.visualize = False
    args.save_images = False
    args.resume = False
    args.savedir = str(output_dir)

    args.weights = str(checkpoint_path)
    args.load = None

    model_path = PROJECT_ROOT / "src" / "sam_tss" / "models" / args.model
    model = load_model_from_file(args, str(model_path), board=None, device=device, checkpoint=None)

    train_count = 0
    test_count = 0

    if cli.num_train > 0:
        train_count = run_split(
            args=args,
            device=device,
            model=model,
            split=cli.train_split,
            n_samples=cli.num_train,
            output_dir=output_dir,
            num_workers=cli.num_workers,
        )

    if cli.num_test > 0:
        test_count = run_split(
            args=args,
            device=device,
            model=model,
            split=cli.test_split,
            n_samples=cli.num_test,
            output_dir=output_dir,
            num_workers=cli.num_workers,
        )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved train samples: {train_count} -> {output_dir / cli.train_split}")
    print(f"Saved test samples:  {test_count} -> {output_dir / cli.test_split}")


if __name__ == "__main__":
    main()
