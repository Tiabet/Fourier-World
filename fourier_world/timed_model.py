"""Fourier-World with timestamp-conditioned frequency keys."""

from __future__ import annotations

from numbers import Real
from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F

from .model import (
    FourierStateQueryBlock,
    FourierWorld,
    _spatial_rope,
    _time_rope,
)
from .time_embedding import (
    FrequencyTimeKeyEmbedding,
    validate_monotonic_times,
)


class TimedFourierStateQueryBlock(FourierStateQueryBlock):
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mode_key: torch.Tensor,
        future_times: torch.Tensor,
        memory_times: torch.Tensor,
        current_memory_indices: torch.Tensor,
        motion_mask: torch.Tensor,
        value_mask: torch.Tensor,
        *,
        spatial_shape: tuple[int, int],
        spatial_coordinates: torch.Tensor | None = None,
        action_scale: float = 1.0,
    ) -> torch.Tensor:
        batch, spatial, hidden = query.shape
        if memory.ndim != 4 or memory.shape[:2] != (batch, spatial):
            raise ValueError("Fourier memory must be [B,S,M,H]")
        modes = memory.shape[2]
        if mode_key.shape == (modes, hidden):
            flat_mode_key = mode_key[None].expand(batch * spatial, -1, -1)
        elif mode_key.shape == (batch, modes, hidden):
            flat_mode_key = (
                mode_key[:, None]
                .expand(-1, spatial, -1, -1)
                .reshape(batch * spatial, modes, hidden)
            )
        else:
            raise ValueError("mode keys must be [M,H] or [B,M,H]")

        flat_query = query.reshape(batch * spatial, 1, hidden)
        flat_memory = memory.reshape(batch * spatial, modes, hidden)
        q = self._temporal_heads(self.temporal_q(self.query_norm(flat_query)))
        if value_mask.shape != (modes,):
            raise ValueError("Fourier value mask must have one entry per memory token")
        expanded_future = future_times[:, None].expand(-1, spatial, -1).reshape(-1, 1)
        expanded_memory = (
            memory_times[:, None].expand(-1, spatial, -1).reshape(batch * spatial, modes)
        )
        q = _time_rope(q, expanded_future, rope_dim=self.time_rope_dim)

        current_memory = flat_memory.index_select(1, current_memory_indices)
        current_key = flat_mode_key.index_select(1, current_memory_indices)
        current_times = expanded_memory.index_select(1, current_memory_indices)
        current_value_mask = value_mask.index_select(0, current_memory_indices)
        key, value = self._project_temporal_kv(current_memory)
        key = key + current_key
        value = value * current_value_mask.to(value.dtype)[None, :, None]
        key = self._temporal_heads(key)
        value = self._temporal_heads(value)
        key = _time_rope(key, current_times, rope_dim=self.time_rope_dim)
        current_read = F.scaled_dot_product_attention(q, key, value)
        current_read = self._merge_temporal_heads(current_read)
        innovation = self.temporal_scale * self.temporal_out(current_read)
        innovation = innovation.reshape(batch, spatial, hidden)
        innovation = innovation * motion_mask.to(innovation.dtype)

        q_spatial, k_spatial = self.spatial_qk(self.spatial_query_norm(query)).chunk(2, dim=-1)
        q_spatial = self._spatial_heads(q_spatial)
        k_spatial = self._spatial_heads(k_spatial)
        height, width = spatial_shape
        q_spatial = _spatial_rope(q_spatial, height, width, spatial_coordinates)
        k_spatial = _spatial_rope(k_spatial, height, width, spatial_coordinates)
        v_spatial = self._spatial_heads(self.spatial_v(self.spatial_innovation_norm(innovation)))
        propagated = F.scaled_dot_product_attention(q_spatial, k_spatial, v_spatial)
        propagated = propagated.transpose(1, 2).reshape(batch, spatial, hidden)
        innovation = innovation + self.spatial_scale * self.spatial_out(propagated)
        innovation = innovation + self.ffn_scale * self.ffn_down(
            F.gelu(self.ffn_up(self.ffn_norm(innovation)))
        )
        innovation = innovation * motion_mask.to(innovation.dtype)
        return query + float(action_scale) * innovation


class TimedFourierStateQueryState(TypedDict):
    tail: torch.Tensor
    tail_times: torch.Tensor
    spectrum: torch.Tensor
    window_scale: torch.Tensor
    window_times: torch.Tensor
    spectrum_times: torch.Tensor


class TimedFourierWorld(FourierWorld):
    """Predict at a requested timestamp using interval-conditioned frequency keys."""

    method_revision = "fourierworld_learned_mask_irregular_time_key_v1"
    core_method_revision = "fourierworld_learned_mask_query"
    timing_revision = "sample_index_rfft4_frequency_time_K_only_v1"

    def __init__(self, *args, timing_hidden: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layer_scale_init = float(self.blocks[0].temporal_scale[0].detach())
        self.blocks = nn.ModuleList(
            TimedFourierStateQueryBlock(
                self.hidden,
                self.heads,
                self.intermediate,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(self.depth)
        )
        self.frequency_time_key = FrequencyTimeKeyEmbedding(
            self.hidden,
            conditioner_hidden=timing_hidden,
        )

    @torch.compiler.disable
    def analyze(self, context, timestamps):
        return super().analyze(context, timestamps)

    @torch.compiler.disable
    def _memory_tokens(
        self,
        spectrum,
        scale,
        window_times,
        *,
        dc_scale,
        ac_scale,
        imaginary_scale,
        nyquist_scale,
    ):
        return super()._memory_tokens(
            spectrum,
            scale,
            window_times,
            dc_scale=dc_scale,
            ac_scale=ac_scale,
            imaginary_scale=imaginary_scale,
            nyquist_scale=nyquist_scale,
        )

    def _predict_timed(
        self,
        latest,
        spectrum,
        scale,
        window_times,
        context_times,
        future_times,
        *,
        spatial_shape,
        spatial_coordinates=None,
        dc_scale=1.0,
        ac_scale=1.0,
        imaginary_scale=1.0,
        nyquist_scale=1.0,
        action_scale=1.0,
        frequency_time_scale=1.0,
    ):
        memory, modes, memory_window_indices, motion_mask = self._memory_tokens(
            spectrum,
            scale,
            window_times,
            dc_scale=dc_scale,
            ac_scale=ac_scale,
            imaginary_scale=imaginary_scale,
            nyquist_scale=nyquist_scale,
        )
        origin = window_times[:, -1:]
        memory_times = (
            window_times.index_select(1, memory_window_indices) - origin
        ) / self.reference_dt
        normalized_future = (future_times - origin) / self.reference_dt
        base_key = self.mode_key.index_select(0, modes)[None].expand(latest.shape[0], -1, -1)
        timing_key = self.frequency_time_key(
            context_times,
            future_times,
            reference_dt=self.reference_dt,
        ).to(base_key.dtype)
        mode_key = base_key + float(frequency_time_scale) * timing_key
        value_mask = modes != 0
        current_memory_indices = torch.arange(3, device=memory.device)
        initial_query = self.query_mask.to(latest.dtype).expand(
            latest.shape[0], latest.shape[1], -1
        )
        query = initial_query
        for block in self.blocks:
            query = block(
                query,
                memory,
                mode_key,
                normalized_future,
                memory_times,
                current_memory_indices,
                motion_mask,
                value_mask,
                spatial_shape=spatial_shape,
                spatial_coordinates=spatial_coordinates,
                action_scale=action_scale,
            )
        return (latest + (query - initial_query)).to(latest.dtype)

    @staticmethod
    def _future_tensor(
        future_timestamp,
        times: torch.Tensor,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if future_timestamp is None:
            return times[:, -1:] + (times[:, -1:] - times[:, -2:-1])
        if isinstance(future_timestamp, Real):
            return torch.full((batch, 1), float(future_timestamp), device=device)
        if not torch.is_tensor(future_timestamp):
            raise TypeError("future_timestamp must be real or tensor")
        future = future_timestamp.to(device, torch.float32)
        if future.ndim == 0:
            future = future.expand(batch, 1)
        elif future.shape == (batch,):
            future = future[:, None]
        return future

    def forward(
        self,
        context: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        *,
        future_timestamp=None,
        spatial_shape=None,
        spatial_coordinates=None,
        dc_scale=1.0,
        ac_scale=1.0,
        imaginary_scale=1.0,
        nyquist_scale=1.0,
        action_scale=1.0,
        frequency_time_scale=1.0,
    ) -> torch.Tensor:
        self._validate_context(context)
        batch, observed_steps, spatial, _ = context.shape
        if timestamps is None:
            context = context[:, -4:]
            times = self._default_times(context)
        else:
            times = validate_monotonic_times(timestamps, batch, observed_steps).to(context.device)[
                :, -4:
            ]
            context = context[:, -4:]
        future = self._future_tensor(
            future_timestamp,
            times,
            batch,
            context.device,
        )
        if future.shape != (batch, 1) or not (future >= times[:, -1:]).all():
            raise ValueError("future timestamp must be [B,1] and not precede context")
        spectrum, scale, window_times = self.analyze(context, times)
        return self._predict_timed(
            context[:, -1],
            spectrum,
            scale,
            window_times,
            times,
            future,
            spatial_shape=self._resolve_spatial_shape(spatial, spatial_shape),
            spatial_coordinates=spatial_coordinates,
            dc_scale=dc_scale,
            ac_scale=ac_scale,
            imaginary_scale=imaginary_scale,
            nyquist_scale=nyquist_scale,
            action_scale=action_scale,
            frequency_time_scale=frequency_time_scale,
        )

    def initialize(
        self,
        context: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> TimedFourierStateQueryState:
        self._validate_context(context)
        if timestamps is None:
            times = self._default_times(context)
        else:
            times = validate_monotonic_times(timestamps, context.shape[0], context.shape[1]).to(
                context.device
            )
        context, times = context[:, -4:], times[:, -4:]
        spectrum, scale, window_times = self.analyze(context, times)
        return {
            "tail": context[:, -3:],
            "tail_times": times[:, -3:],
            "spectrum": spectrum,
            "window_scale": scale,
            "window_times": window_times,
            "spectrum_times": times,
        }

    def step(
        self,
        state: TimedFourierStateQueryState,
        *,
        future_timestamp=None,
        spatial_shape=None,
        spatial_coordinates=None,
        **interventions,
    ) -> tuple[torch.Tensor, TimedFourierStateQueryState]:
        tail, tail_times = state["tail"], state["tail_times"]
        batch, _, spatial, _ = tail.shape
        future = self._future_tensor(
            future_timestamp,
            tail_times,
            batch,
            tail.device,
        )
        if future.shape != (batch, 1) or not (future >= tail_times[:, -1:]).all():
            raise ValueError("future timestamp must not precede the recurrent tail")
        prediction = self._predict_timed(
            tail[:, -1],
            state["spectrum"],
            state["window_scale"],
            state["window_times"],
            state["spectrum_times"],
            future,
            spatial_shape=self._resolve_spatial_shape(spatial, spatial_shape),
            spatial_coordinates=spatial_coordinates,
            **interventions,
        )
        new_window = torch.cat((tail, prediction[:, None]), dim=1)
        new_times = torch.cat((tail_times, future), dim=1)
        new_spectrum, new_scale, new_window_times = self.analyze(new_window, new_times)
        next_state: TimedFourierStateQueryState = {
            "tail": new_window[:, 1:],
            "tail_times": new_times[:, 1:],
            "spectrum": new_spectrum,
            "window_scale": new_scale,
            "window_times": new_window_times,
            "spectrum_times": new_times,
        }
        return prediction, next_state

    def rollout(
        self,
        context: torch.Tensor,
        steps: int,
        timestamps: torch.Tensor | None = None,
        *,
        spatial_shape=None,
        spatial_coordinates=None,
        **interventions,
    ) -> list[torch.Tensor]:
        if int(steps) < 1:
            raise ValueError("steps must be positive")
        state = self.initialize(
            context[:, -4:], timestamps if timestamps is None else timestamps[:, -4:]
        )
        outputs = []
        for _ in range(int(steps)):
            prediction, state = self.step(
                state,
                spatial_shape=spatial_shape,
                spatial_coordinates=spatial_coordinates,
                **interventions,
            )
            outputs.append(prediction)
        return outputs
