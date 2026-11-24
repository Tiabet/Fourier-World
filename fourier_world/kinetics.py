"""Streaming Kinetics-700 video sampling and shard-prefix validation."""

from __future__ import annotations

import random
import tarfile
import io
import math
from pathlib import Path
from typing import Iterator

import cv2
import decord
import numpy as np
import torch
from decord import DECORDError
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info


DATA_ERRORS = (
    OSError,
    ValueError,
    RuntimeError,
    cv2.error,
    DECORDError,
    tarfile.TarError,
)


def sample_irregular_indices(
    timestamps: torch.Tensor,
    *,
    num_frames: int = 8,
    time_stride_range: tuple[float, float] = (1 / 25, 1 / 3),
    generator: torch.Generator | None = None,
) -> tuple[list[int], torch.Tensor, int]:
    if timestamps.ndim != 1 or timestamps.numel() < num_frames:
        raise ValueError("video timestamps must be a sufficiently long vector")
    if not torch.isfinite(timestamps).all() or not (timestamps[1:] > timestamps[:-1]).all():
        raise ValueError("video timestamps must be finite and strictly increasing")
    minimum, maximum = map(float, time_stride_range)
    if not 0 < minimum <= maximum:
        raise ValueError("invalid time-stride range")
    deltas = torch.rand(num_frames - 1, generator=generator) * (maximum - minimum) + minimum
    cumulative = torch.cat((torch.zeros(1), deltas.cumsum(0)))
    maximum_start = max(0.0, timestamps[-1].item() - cumulative[-1].item())
    target = torch.rand(1, generator=generator).item() * maximum_start + cumulative
    indices = (timestamps.unsqueeze(0) - target.unsqueeze(1)).abs().argmin(dim=1).tolist()
    duplicate_adjacencies = sum(left == right for left, right in zip(indices, indices[1:]))
    return indices, timestamps[indices], duplicate_adjacencies


def shared_random_resized_crop(
    frames: np.ndarray,
    *,
    frame_size: int,
    ratio_jitter: float = 4 / 3,
    scale: tuple[float, float] = (0.6, 1.0),
    horizontal_flip: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frame_size < 1:
        raise ValueError("frames must be uint8 [T,H,W,3]")
    height, width = frames.shape[1:3]
    ratio = (1.0 / ratio_jitter, ratio_jitter)
    area = height * width
    log_ratio = torch.log(torch.tensor(ratio))
    crop = None
    for _ in range(10):
        target_area = area * float(
            torch.empty(1).uniform_(
                scale[0],
                scale[1],
                generator=generator,
            )
        )
        aspect_ratio = math.exp(
            float(
                torch.empty(1).uniform_(
                    float(log_ratio[0]),
                    float(log_ratio[1]),
                    generator=generator,
                )
            )
        )
        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            top = int(
                torch.randint(
                    0,
                    height - crop_h + 1,
                    (1,),
                    generator=generator,
                )
            )
            left = int(
                torch.randint(
                    0,
                    width - crop_w + 1,
                    (1,),
                    generator=generator,
                )
            )
            crop = (top, left, crop_h, crop_w)
            break
    if crop is None:
        input_ratio = width / height
        if input_ratio < min(ratio):
            crop_w = width
            crop_h = int(round(crop_w / min(ratio)))
        elif input_ratio > max(ratio):
            crop_h = height
            crop_w = int(round(crop_h * max(ratio)))
        else:
            crop_w, crop_h = width, height
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
    batch = np.stack(
        [
            cv2.resize(
                frame[top : top + crop_h, left : left + crop_w],
                (frame_size, frame_size),
                interpolation=cv2.INTER_LINEAR,
            )
            for frame in frames
        ]
    )
    if horizontal_flip and float(torch.rand(1, generator=generator)) < 0.5:
        batch = batch[:, :, ::-1].copy()
    return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()


def decode_k700_member(
    data: bytes,
    *,
    frame_size: int,
    generator: torch.Generator,
    num_frames: int = 8,
    sample_one_window: bool = False,
    window_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reader = decord.VideoReader(io.BytesIO(data), num_threads=1)
    count = len(reader)
    fps = float(reader.get_avg_fps())
    if int(num_frames) < 5:
        raise ValueError("num_frames must contain at least context-four plus target")
    if not np.isfinite(fps) or not 1.0 <= fps <= 120.0 or count < int(num_frames):
        raise RuntimeError(f"invalid video metadata fps={fps}, frames={count}")

    video_times = torch.from_numpy(
        np.asarray(
            reader.get_frame_timestamp(range(count)),
            dtype=np.float32,
        )[:, 0].copy()
    )
    indices, selected_times, duplicate_adjacencies = sample_irregular_indices(
        video_times,
        num_frames=int(num_frames),
        generator=generator,
    )
    if sample_one_window:
        if int(num_frames) != 8:
            raise ValueError("selected context-four windows require num_frames=8")
        if window_index is None:
            window_index = int(torch.randint(4, (1,), generator=generator).item())
        if not 0 <= int(window_index) < 4:
            raise ValueError("window_index must select one of four context-4 views")
        start = int(window_index)
        indices = indices[start : start + 5]
        selected_times = selected_times[start : start + 5]
    elif window_index is not None:
        raise ValueError("window_index requires sample_one_window")
    frames = reader.get_batch(indices).asnumpy()
    augmented = shared_random_resized_crop(
        frames,
        frame_size=frame_size,
        generator=generator,
    )
    selected_times = selected_times - selected_times[0]
    metadata = torch.tensor((fps, float(duplicate_adjacencies)), dtype=torch.float32)
    return augmented, selected_times, metadata


class K700IrregularTarDataset(IterableDataset):
    """Stream disjoint shard partitions across training ranks and workers."""

    def __init__(
        self,
        root: str | Path,
        *,
        frame_size: int = 512,
        seed: int = 0,
        shuffle_buffer: int = 8,
        maximum_member_bytes: int = 256 * 1024 * 1024,
        exclude_list: str | Path | None = None,
        sample_one_window: bool = False,
        holdout_prefix_members_per_shard: int = 0,
        num_frames: int = 8,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        # A spawned DataLoader worker does not inherit the parent's process group.
        distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if distributed else 0
        self.world_size = dist.get_world_size() if distributed else 1
        self.frame_size = int(frame_size)
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        self.maximum_member_bytes = int(maximum_member_bytes)
        self.sample_one_window = bool(sample_one_window)
        self.num_frames = int(num_frames)
        self.holdout_prefix_members_per_shard = int(holdout_prefix_members_per_shard)
        if self.frame_size < 1 or self.shuffle_buffer < 1 or self.num_frames < 5:
            raise ValueError("frame size and shuffle buffer must be positive")
        if self.sample_one_window and self.num_frames != 8:
            raise ValueError("sample_one_window is defined only for the public eight-frame draw")
        if self.holdout_prefix_members_per_shard < 0:
            raise ValueError("holdout prefix must be nonnegative")
        self.shards = sorted(self.root.glob("k700_train_*.tar.gz"))
        expected = [f"k700_train_{index:03d}.tar.gz" for index in range(1, 701)]
        if [path.name for path in self.shards] != expected:
            raise FileNotFoundError("K700 requires exact shards 001 through 700")
        self.excluded: set[str] = set()
        if exclude_list is not None:
            self.excluded = {
                line.strip().removesuffix(".mp4")
                for line in Path(exclude_list).read_text().splitlines()
                if line.strip()
            }

    def _distributed_worker(self) -> tuple[int, int]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        workers = worker.num_workers if worker is not None else 1
        return self.rank * workers + worker_id, self.world_size * workers

    def _members(
        self,
        shard: Path,
        rng: random.Random,
    ) -> Iterator[tuple[str, bytes]]:
        buffer: list[tuple[str, bytes]] = []
        eligible_index = 0
        with tarfile.open(shard, mode="r|gz") as archive:
            for member in archive:
                key = member.name.removesuffix(".mp4")
                if (
                    not member.isfile()
                    or not member.name.lower().endswith(".mp4")
                    or member.size <= 0
                    or member.size > self.maximum_member_bytes
                    or key in self.excluded
                ):
                    continue
                eligible_index += 1
                if eligible_index <= self.holdout_prefix_members_per_shard:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                buffer.append((member.name, stream.read()))
                if len(buffer) >= self.shuffle_buffer:
                    yield buffer.pop(rng.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))

    def __iter__(self):
        worker_id, workers = self._distributed_worker()
        torch_rng = torch.Generator().manual_seed(self.seed + 1_000_003 * worker_id)
        cycle = 0
        while True:
            shard_rng = random.Random(self.seed + cycle)
            shards = list(self.shards)
            shard_rng.shuffle(shards)
            assigned = shards[worker_id::workers]
            if not assigned:
                raise RuntimeError("more distributed workers than K700 shards")
            for shard in assigned:
                member_rng = random.Random(
                    self.seed
                    + cycle * 10_007
                    + int(shard.name.removeprefix("k700_train_").split(".")[0])
                )
                for name, data in self._members(shard, member_rng):
                    try:
                        frames, timestamps, metadata = decode_k700_member(
                            data,
                            frame_size=self.frame_size,
                            generator=torch_rng,
                            num_frames=self.num_frames,
                            sample_one_window=self.sample_one_window,
                        )
                        yield {
                            "frames": frames,
                            "timestamps": timestamps,
                            "sampling": metadata,
                            "source": name,
                        }
                    except DATA_ERRORS:
                        continue
            cycle += 1


class K700ValidationPrefixDataset(Dataset):
    """Decode one validation clip per shard from the reserved member prefix."""

    def __init__(
        self,
        root: str | Path,
        *,
        frame_size: int = 512,
        seed: int = 17_071,
        prefix_members_per_shard: int = 8,
        maximum_member_bytes: int = 256 * 1024 * 1024,
        exclude_list: str | Path | None = None,
        sample_one_window: bool = True,
        num_frames: int = 8,
    ) -> None:
        self.root = Path(root)
        self.frame_size = int(frame_size)
        self.seed = int(seed)
        self.prefix_members_per_shard = int(prefix_members_per_shard)
        self.maximum_member_bytes = int(maximum_member_bytes)
        self.sample_one_window = bool(sample_one_window)
        self.num_frames = int(num_frames)
        if self.frame_size < 1 or self.prefix_members_per_shard < 1 or self.num_frames < 5:
            raise ValueError("validation frame size and prefix must be positive")
        if self.sample_one_window and self.num_frames != 8:
            raise ValueError("sample_one_window is defined only for the public eight-frame draw")
        self.shards = sorted(self.root.glob("k700_train_*.tar.gz"))
        expected = [f"k700_train_{index:03d}.tar.gz" for index in range(1, 701)]
        if [path.name for path in self.shards] != expected:
            raise FileNotFoundError("K700 validation requires exact shards 001 through 700")
        self.excluded: set[str] = set()
        if exclude_list is not None:
            self.excluded = {
                line.strip().removesuffix(".mp4")
                for line in Path(exclude_list).read_text().splitlines()
                if line.strip()
            }

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, index: int) -> dict:
        shard = self.shards[int(index)]
        eligible = 0
        failures: list[str] = []
        with tarfile.open(shard, mode="r|gz") as archive:
            for member in archive:
                key = member.name.removesuffix(".mp4")
                if (
                    not member.isfile()
                    or not member.name.lower().endswith(".mp4")
                    or member.size <= 0
                    or member.size > self.maximum_member_bytes
                    or key in self.excluded
                ):
                    continue
                eligible += 1
                if eligible > self.prefix_members_per_shard:
                    break
                stream = archive.extractfile(member)
                if stream is None:
                    failures.append(f"{member.name}: no stream")
                    continue
                generator = torch.Generator().manual_seed(
                    self.seed + 1_000_003 * int(index) + 10_007 * eligible
                )
                try:
                    frames, timestamps, metadata = decode_k700_member(
                        stream.read(),
                        frame_size=self.frame_size,
                        generator=generator,
                        num_frames=self.num_frames,
                        sample_one_window=self.sample_one_window,
                    )
                    return {
                        "frames": frames,
                        "timestamps": timestamps,
                        "sampling": metadata,
                        "source": f"{shard.name}:{member.name}",
                        "shard_index": torch.tensor(int(index)),
                        "prefix_index": torch.tensor(eligible - 1),
                    }
                except DATA_ERRORS as error:
                    failures.append(f"{member.name}: {type(error).__name__}")
        raise RuntimeError(
            f"no valid validation member in {shard.name} prefix ({'; '.join(failures)})"
        )
