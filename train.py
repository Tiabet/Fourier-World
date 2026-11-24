"""Train and validate Fourier-World on Kinetics-700 video shards."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from fourier_world.backbone import DINOv3DenseFeatures, file_sha256
from fourier_world.kinetics import K700IrregularTarDataset, K700ValidationPrefixDataset
from fourier_world.windows import expand_sliding_context4
from fourier_world.timed_model import TimedFourierWorld


def grouped_adamw(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay_types = (
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
    )
    decay = {}
    for module_name, module in model.named_modules():
        if isinstance(module, decay_types):
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.nn.Parameter) and weight.requires_grad:
                decay[f"{module_name}.weight" if module_name else "weight"] = weight
    no_decay = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in decay
    }
    return torch.optim.AdamW(
        [
            {"params": [decay[name] for name in sorted(decay)], "weight_decay": weight_decay},
            {"params": [no_decay[name] for name in sorted(no_decay)], "weight_decay": 0.0},
        ],
        lr=lr,
    )


def microbatch_plan(global_batch: int, microbatch: int, world_size: int) -> tuple[int, ...]:
    if min(global_batch, microbatch, world_size) < 1 or global_batch % world_size:
        raise ValueError("invalid global batch, microbatch or world size")
    local_batch = global_batch // world_size
    full, remainder = divmod(local_batch, microbatch)
    plan = (microbatch,) * full + ((remainder,) if remainder else ())
    if not plan or sum(plan) * world_size != global_batch:
        raise ValueError("microbatch plan does not preserve the global batch")
    return plan


def learning_rate(
    step: int, base: float, warmup: int, cooldown_start: int, cooldown: float
) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    return base if step < cooldown_start else cooldown


def transition_loss(
    model: torch.nn.Module,
    features: torch.Tensor,
    timestamps: torch.Tensor,
) -> torch.Tensor:
    contexts, context_times, future_times, targets = expand_sliding_context4(
        features,
        timestamps,
        windows_per_clip=1,
    )
    side = math.isqrt(features.shape[2])
    if side * side != features.shape[2]:
        raise ValueError("the DINO feature grid must be square")
    prediction = model(
        contexts,
        context_times,
        future_timestamp=future_times,
        spatial_shape=(side, side),
    )
    return F.smooth_l1_loss(prediction.float(), targets.float(), beta=0.1)


def atomic_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def encode(
    backbone: torch.nn.Module,
    frames: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        return backbone(frames.to(device, non_blocking=True)).detach()


def validation_loader(
    tars: Path,
    *,
    frame_size: int,
    seed: int,
    holdout_prefix: int,
    exclude_list: Path | None,
    samples: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    if samples < 1 or samples > 700 or samples % world_size:
        raise ValueError("validation samples must be between 1 and 700 and divide across ranks")
    validation = K700ValidationPrefixDataset(
        tars,
        frame_size=frame_size,
        seed=seed + 91_177,
        prefix_members_per_shard=holdout_prefix,
        exclude_list=exclude_list,
        sample_one_window=False,
    )
    indices = np.rint(np.linspace(0, len(validation) - 1, samples)).astype(np.int64).tolist()
    if len(set(indices)) != samples:
        raise ValueError("validation shard selection is not unique")
    return DataLoader(
        Subset(validation, indices[rank::world_size]),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )


@torch.inference_mode()
def validate(
    model: TimedFourierWorld,
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed: bool,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    for batch in loader:
        features = encode(backbone, batch["frames"], device)
        times = batch["timestamps"].to(device, non_blocking=True)
        if features.shape[1] != 8:
            raise ValueError("validation clips must have eight sampled frames")
        side = math.isqrt(features.shape[2])
        if side * side != features.shape[2]:
            raise ValueError("the DINO feature grid must be square")
        context = features[:, :4]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            h1 = model(
                context, times[:, :4], future_timestamp=times[:, 4:5], spatial_shape=(side, side)
            )
            h2 = model(
                torch.cat((context[:, 1:], h1[:, None]), dim=1),
                times[:, 1:5],
                future_timestamp=times[:, 5:6],
                spatial_shape=(side, side),
            )
            h3 = model(
                torch.cat((context[:, 2:], h1[:, None], h2[:, None]), dim=1),
                times[:, 2:6],
                future_timestamp=times[:, 6:7],
                spatial_shape=(side, side),
            )
        copy = context[:, -1].float()
        totals[0] += (
            F.smooth_l1_loss(h1.float(), features[:, 4].float(), beta=0.1, reduction="none")
            .mean((1, 2))
            .sum()
        )
        totals[1] += (
            F.smooth_l1_loss(h3.float(), features[:, 6].float(), beta=0.1, reduction="none")
            .mean((1, 2))
            .sum()
        )
        totals[2] += (
            F.smooth_l1_loss(copy, features[:, 4].float(), beta=0.1, reduction="none")
            .mean((1, 2))
            .sum()
        )
        totals[3] += (
            F.smooth_l1_loss(copy, features[:, 6].float(), beta=0.1, reduction="none")
            .mean((1, 2))
            .sum()
        )
        totals[4] += features.shape[0]
    if distributed:
        dist.all_reduce(totals)
    if totals[4] < 1:
        raise ValueError("empty validation population")
    model.train(was_training)
    return {
        "h1_smooth_l1": float(totals[0] / totals[4]),
        "h3_smooth_l1": float(totals[1] / totals[4]),
        "copy_h1_smooth_l1": float(totals[2] / totals[4]),
        "copy_h3_smooth_l1": float(totals[3] / totals[4]),
        "samples": int(totals[4]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tars", type=Path, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-list", type=Path)
    parser.add_argument("--backbone-sha256")
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--global-batch", type=int, default=1024)
    parser.add_argument("--microbatch", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--val-workers", type=int, default=2)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--holdout-prefix", type=int, default=8)
    parser.add_argument("--shuffle-buffer", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--cooldown-start", type=int, default=55_000)
    parser.add_argument("--cooldown-lr", type=float, default=1e-5)
    parser.add_argument("--reference-dt", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compile-predictor", action="store_true")
    parser.add_argument("--compile-backbone", action="store_true")
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if (
        min(
            args.steps,
            args.frame_size,
            args.holdout_prefix,
            args.val_samples,
            args.save_every,
            args.log_every,
        )
        < 1
    ):
        raise ValueError("training and validation sizes must be positive")
    if min(args.workers, args.val_workers, args.val_every) < 0:
        raise ValueError("worker counts and validation cadence must be nonnegative")
    if args.val_samples > 700 or args.val_samples % world_size:
        raise ValueError("validation samples must be at most 700 and divide across ranks")
    if not 0 < args.warmup_steps < args.cooldown_start <= args.steps:
        raise ValueError("invalid learning-rate schedule")
    plan = microbatch_plan(args.global_batch, args.microbatch, world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    random.seed(args.seed + rank)
    np.random.seed((args.seed + rank) % (2**32))
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pt").exists():
        raise FileExistsError(args.output / "last.pt")
    backbone = DINOv3DenseFeatures(args.backbone)
    if args.backbone_sha256 is not None and backbone.weights_sha256 != args.backbone_sha256:
        raise ValueError("DINOv3 backbone hash mismatch")
    validation_config = {
        "frame_size": args.frame_size,
        "seed": args.seed,
        "holdout_prefix": args.holdout_prefix,
        "val_samples": args.val_samples,
        "reference_dt": args.reference_dt,
        "backbone_sha256": backbone.weights_sha256,
        "exclude_list_sha256": file_sha256(args.exclude_list) if args.exclude_list else None,
    }
    backbone.eval().requires_grad_(False).to(device)
    model = TimedFourierWorld(reference_dt=args.reference_dt, timing_hidden=16).to(device)
    if model.active_parameter_count != 85_665_520:
        raise ValueError("unexpected FourierWorld parameter count")
    optimizer = grouped_adamw(model, args.lr, args.weight_decay)
    step = 0
    resume = args.resume or (
        args.output / "progress.pt" if (args.output / "progress.pt").exists() else None
    )
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        saved_config = checkpoint.get("validation_config", {})
        for field, value in saved_config.items():
            if field in validation_config and validation_config[field] != value:
                raise ValueError(f"resume settings differ for {field}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        if checkpoint["global_batch"] != args.global_batch:
            raise ValueError("resume global batch mismatch")
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint
    if step >= args.steps:
        raise ValueError("resume checkpoint has already reached the requested training length")
    train_data = K700IrregularTarDataset(
        args.tars,
        frame_size=args.frame_size,
        seed=args.seed + 1_000_003 * step,
        shuffle_buffer=args.shuffle_buffer,
        exclude_list=args.exclude_list,
        sample_one_window=True,
        holdout_prefix_members_per_shard=args.holdout_prefix,
    )
    loader = DataLoader(
        train_data,
        batch_size=args.microbatch,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    iterator = iter(loader)
    val_loader = validation_loader(
        args.tars,
        frame_size=args.frame_size,
        seed=args.seed,
        holdout_prefix=args.holdout_prefix,
        exclude_list=args.exclude_list,
        samples=args.val_samples,
        batch_size=args.microbatch,
        workers=args.val_workers,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    train_backbone = torch.compile(backbone) if args.compile_backbone else backbone
    train_model = torch.compile(model) if args.compile_predictor else model
    if distributed:
        train_model = DistributedDataParallel(
            train_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    model.train()
    local_batch = args.global_batch // world_size
    while step < args.steps:
        lr = learning_rate(step, args.lr, args.warmup_steps, args.cooldown_start, args.cooldown_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        local_loss = 0.0
        for index, size in enumerate(plan):
            batch = next(iterator)
            frames = batch["frames"][:size]
            times = batch["timestamps"][:size].to(device, non_blocking=True)
            features = encode(train_backbone, frames, device)
            context = (
                train_model.no_sync() if distributed and index < len(plan) - 1 else nullcontext()
            )
            with context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    loss = transition_loss(train_model, features, times)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite FourierWorld loss")
                (loss * size / local_batch).backward()
            local_loss += float(loss.detach()) * size
        optimizer.step()
        step += 1
        if step == 1 or step % args.log_every == 0:
            totals = torch.tensor((local_loss, local_batch), dtype=torch.float64, device=device)
            if distributed:
                dist.all_reduce(totals)
            if rank == 0:
                print(
                    json.dumps({"step": step, "loss": float(totals[0] / totals[1]), "lr": lr}),
                    flush=True,
                )
        if args.val_every and step % args.val_every == 0:
            metrics = validate(model, backbone, val_loader, device, distributed)
            if rank == 0:
                print(json.dumps({"step": step, "validation": metrics}), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                atomic_save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "global_batch": args.global_batch,
                        "validation_config": validation_config,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else None,
                    },
                    args.output / "progress.pt",
                )
            if distributed:
                dist.barrier()
    if rank == 0:
        os.replace(args.output / "progress.pt", args.output / "last.pt")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
