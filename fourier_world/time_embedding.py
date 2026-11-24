"""Physical-time features for irregular video sampling."""

from __future__ import annotations

import torch
from torch import nn


def validate_monotonic_times(
    times: torch.Tensor,
    batch: int,
    steps: int,
) -> torch.Tensor:
    value = times.float()
    if value.shape != (batch, steps):
        raise ValueError(f"timestamps must be [{batch},{steps}], got {tuple(value.shape)}")
    if not torch.isfinite(value).all() or not (value[:, 1:] >= value[:, :-1]).all():
        raise ValueError("timestamps must be finite and nondecreasing")
    return value


class FrequencyTimeKeyEmbedding(nn.Module):
    def __init__(self, hidden: int, conditioner_hidden: int = 16) -> None:
        super().__init__()
        if hidden < 1 or conditioner_hidden < 1:
            raise ValueError("frequency/time conditioner widths must be positive")
        self.net = nn.Sequential(
            nn.Linear(7, conditioner_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(conditioner_hidden, hidden, bias=False),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(
        self,
        context_times: torch.Tensor,
        future_times: torch.Tensor,
        *,
        reference_dt: float,
    ) -> torch.Tensor:
        if context_times.ndim != 2 or context_times.shape[1] != 4:
            raise ValueError("frequency/time conditioning needs [B,4] context times")
        if future_times.shape != (context_times.shape[0], 1):
            raise ValueError("future times must be [B,1]")
        intervals = torch.cat(
            (
                context_times[:, 1:] - context_times[:, :-1],
                future_times - context_times[:, -1:],
            ),
            dim=1,
        ).float()
        if not torch.isfinite(intervals).all() or not (intervals >= 0).all():
            raise ValueError("all observed and future intervals must be finite and nonnegative")
        ratio = intervals / float(reference_dt)
        log_ratio = torch.where(
            (ratio - 1.0).abs() <= 1e-6,
            torch.zeros_like(ratio),
            torch.log1p(ratio)
            - torch.log(
                torch.tensor(
                    2.0,
                    device=ratio.device,
                    dtype=ratio.dtype,
                )
            ),
        )
        mean = log_ratio.mean(dim=1, keepdim=True)
        irregularity = log_ratio.std(dim=1, keepdim=True, unbiased=False)
        mode = (
            torch.tensor(
                (0.0, 0.5, 1.0),
                device=log_ratio.device,
                dtype=log_ratio.dtype,
            )
            .reshape(1, 3, 1)
            .expand(log_ratio.shape[0], -1, -1)
        )
        gaps = log_ratio[:, None].expand(-1, 3, -1)
        features = torch.cat(
            (
                gaps,
                mode * mean[:, None],
                mode.square() * mean[:, None],
                mode * irregularity[:, None],
            ),
            dim=-1,
        )
        return self.net(features)
