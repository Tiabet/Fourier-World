"""Build four-frame prediction windows and align their targets."""

from __future__ import annotations

import torch


def expand_sliding_context4(
    features: torch.Tensor,
    timestamps: torch.Tensor,
    *,
    windows_per_clip: int | None = None,
    window_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if features.ndim != 4 or features.shape[1] < 5:
        raise ValueError("features must be [B,T>=5,S,D]")
    if timestamps.shape != features.shape[:2]:
        raise ValueError("timestamps must be [B,T]")
    transitions = features.shape[1] - 4
    selected = transitions if windows_per_clip is None else int(windows_per_clip)
    if selected not in (1, transitions):
        raise ValueError("windows_per_clip must be one or all available transitions")
    contexts = torch.stack([features[:, start : start + 4] for start in range(transitions)], dim=1)
    context_times = torch.stack(
        [timestamps[:, start : start + 4] for start in range(transitions)], dim=1
    )
    future_times = timestamps[:, 4:]
    targets = features[:, 4:]
    if selected == 1:
        batch = features.shape[0]
        if window_indices is None:
            window_indices = torch.randint(transitions, (batch,), device=features.device)
        else:
            window_indices = window_indices.to(features.device)
        if window_indices.shape != (batch,) or not (
            (window_indices >= 0).all() and (window_indices < transitions).all()
        ):
            raise ValueError("window_indices must select one valid transition per clip")
        rows = torch.arange(batch, device=features.device)
        contexts = contexts[rows, window_indices][:, None]
        context_times = context_times[rows, window_indices][:, None]
        future_times = future_times[rows, window_indices][:, None]
        targets = targets[rows, window_indices][:, None]
    contexts = contexts.flatten(0, 1)
    context_times = context_times.flatten(0, 1)
    future_times = future_times.flatten(0, 1)
    targets = targets.flatten(0, 1)
    return contexts, context_times, future_times, targets
