"""Evaluate a Fourier-World checkpoint on held-out Kinetics-700 clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fourier_world import TimedFourierWorld
from fourier_world.backbone import DINOv3DenseFeatures, file_sha256
from train import validate, validation_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tars", required=True, type=Path)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--exclude-list", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--samples", type=int, help="Override the checkpoint's validation sample count."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = checkpoint.get("validation_config", {})
    exclusion_hash = file_sha256(args.exclude_list) if args.exclude_list else None
    if "exclude_list_sha256" in config and config["exclude_list_sha256"] != exclusion_hash:
        raise ValueError("provide the same exclusion list used to train this checkpoint")
    model = TimedFourierWorld(reference_dt=config.get("reference_dt", 0.2))
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval().to(device)
    backbone = DINOv3DenseFeatures(args.backbone)
    if config.get("backbone_sha256") and backbone.weights_sha256 != config["backbone_sha256"]:
        raise ValueError("DINOv3 backbone weights differ from the training checkpoint")
    backbone.eval().requires_grad_(False).to(device)
    loader = validation_loader(
        args.tars,
        frame_size=config.get("frame_size", 512),
        seed=config.get("seed", 0),
        holdout_prefix=config.get("holdout_prefix", 8),
        exclude_list=args.exclude_list,
        samples=config.get("val_samples", 256) if args.samples is None else args.samples,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    print(json.dumps(validate(model, backbone, loader, device, distributed=False)), flush=True)


if __name__ == "__main__":
    main()
