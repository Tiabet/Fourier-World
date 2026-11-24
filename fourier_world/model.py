"""Fourier-World feature predictor for regularly sampled observations."""

from __future__ import annotations

import math
from numbers import Real
from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F


def _validate_times(times: torch.Tensor, batch: int, steps: int) -> torch.Tensor:
    value = times.float()
    if value.shape != (batch, steps):
        raise ValueError(f"timestamps must be [{batch},{steps}], got {tuple(value.shape)}")
    if not torch.isfinite(value).all() or not (value[:, 1:] > value[:, :-1]).all():
        raise ValueError("timestamps must be finite and strictly increasing")
    if steps >= 3:
        intervals = value[:, 1:] - value[:, :-1]
        tolerance = 1e-4 * intervals[:, :1].abs().clamp_min(1e-6)
        if not ((intervals - intervals[:, :1]).abs() <= tolerance).all():
            raise ValueError("ordinary rFFT requires uniformly sampled observations")
    return value


def _time_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_dim: int,
) -> torch.Tensor:
    if rope_dim == 0:
        return tensor
    half = rope_dim // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=tensor.device, dtype=torch.float32)
        / max(1, half - 1)
    )
    angle = positions.float()[:, None, :, None] * frequency[None, None, None]
    cosine, sine = angle.cos().to(tensor.dtype), angle.sin().to(tensor.dtype)
    first, second = tensor[..., :half], tensor[..., half:rope_dim]
    rotated = torch.cat(
        (
            first * cosine - second * sine,
            first * sine + second * cosine,
        ),
        dim=-1,
    )
    return torch.cat((rotated, tensor[..., rope_dim:]), dim=-1)


def _spatial_rope(
    tensor: torch.Tensor,
    height: int,
    width: int,
    coordinates: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, heads, tokens, head_dim = tensor.shape
    if height * width != tokens or head_dim % 4:
        raise ValueError("2-D RoPE requires H*W tokens and head dimension divisible by four")
    axis_dim = head_dim // 2
    half = axis_dim // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=tensor.device, dtype=torch.float32)
        / max(1, half - 1)
    )
    if coordinates is None:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=tensor.device, dtype=torch.float32),
            torch.arange(width, device=tensor.device, dtype=torch.float32),
            indexing="ij",
        )
        xx, yy = xx.reshape(-1), yy.reshape(-1)
    else:
        coordinates = coordinates.to(device=tensor.device, dtype=torch.float32)
        if coordinates.shape != (tokens, 2) or not torch.isfinite(coordinates).all():
            raise ValueError("spatial coordinates must be finite [tokens,2] x/y pairs")
        xx, yy = coordinates.unbind(dim=-1)

    def rotate(value: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        first, second = value[..., :half], value[..., half:axis_dim]
        angle = position.reshape(1, 1, tokens, 1) * frequency.reshape(1, 1, 1, half)
        cosine, sine = angle.cos().to(value.dtype), angle.sin().to(value.dtype)
        return torch.cat(
            (
                first * cosine - second * sine,
                first * sine + second * cosine,
            ),
            dim=-1,
        )

    x_part, y_part = tensor[..., :axis_dim], tensor[..., axis_dim : 2 * axis_dim]
    return torch.cat((rotate(x_part, xx), rotate(y_part, yy)), dim=-1).reshape(
        batch, heads, tokens, head_dim
    )


class FourierQueryState(TypedDict):
    tail: torch.Tensor
    tail_times: torch.Tensor
    spectrum: torch.Tensor
    window_scale: torch.Tensor
    window_times: torch.Tensor


class ComplexModeLift(nn.Module):
    """Map the real and imaginary components with a shared complex linear transform."""

    def __init__(self, feature_dim: int, hidden: int) -> None:
        super().__init__()
        if hidden % 2:
            raise ValueError("complex mode hidden size must be even")
        half = hidden // 2
        self.real = nn.Linear(feature_dim, half, bias=False)
        self.imag = nn.Linear(feature_dim, half, bias=False)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        out_real = self.real(real) - self.imag(imag)
        out_imag = self.imag(real) + self.real(imag)
        return torch.cat((out_real, out_imag), dim=-1)


class ComplexHiddenLinear(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        if hidden % 2:
            raise ValueError("complex hidden size must be even")
        half = hidden // 2
        self.real = nn.Linear(half, half, bias=False)
        self.imag = nn.Linear(half, half, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        real, imag = value.chunk(2, dim=-1)
        out_real = self.real(real) - self.imag(imag)
        out_imag = self.imag(real) + self.real(imag)
        return torch.cat((out_real, out_imag), dim=-1)


class PairedComplexRMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden % 2 or eps <= 0:
            raise ValueError("complex RMSNorm needs an even width and positive eps")
        self.hidden = int(hidden)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden // 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.hidden:
            raise ValueError(f"expected complex width {self.hidden}, got {value.shape[-1]}")
        inverse_rms = torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + self.eps).to(
            value.dtype
        )
        paired_weight = torch.cat((self.weight, self.weight)).to(value.dtype)
        return value * inverse_rms * paired_weight


class FourierStateQueryBlock(nn.Module):
    """Read temporal Fourier values per patch, then propagate the update spatially."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        intermediate: int,
        *,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden % heads or (hidden // heads) % 4:
            raise ValueError("hidden/head shape must support 2-D RoPE")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.time_rope_dim = self.head_dim - self.head_dim % 2

        self.query_norm = nn.RMSNorm(hidden)
        self.memory_norm = PairedComplexRMSNorm(hidden)
        self.temporal_q = nn.Linear(hidden, hidden, bias=False)

        self.temporal_k = ComplexHiddenLinear(hidden)
        self.temporal_v = ComplexHiddenLinear(hidden)
        self.temporal_out = nn.Linear(hidden, hidden, bias=False)

        self.spatial_query_norm = nn.RMSNorm(hidden)
        self.spatial_innovation_norm = nn.RMSNorm(hidden)
        self.spatial_qk = nn.Linear(hidden, 2 * hidden, bias=False)
        self.spatial_v = nn.Linear(hidden, hidden, bias=False)
        self.spatial_out = nn.Linear(hidden, hidden, bias=False)

        self.ffn_norm = nn.RMSNorm(hidden)
        self.ffn_up = nn.Linear(hidden, intermediate, bias=False)
        self.ffn_down = nn.Linear(intermediate, hidden, bias=False)

        self.temporal_scale = nn.Parameter(torch.full((hidden,), float(layer_scale_init)))
        self.spatial_scale = nn.Parameter(torch.full((hidden,), float(layer_scale_init)))
        self.ffn_scale = nn.Parameter(torch.full((hidden,), float(layer_scale_init)))

    def _spatial_heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim).transpose(
            1, 2
        )

    def _temporal_heads(self, value: torch.Tensor) -> torch.Tensor:
        real, imag = value.chunk(2, dim=-1)
        complex_head_dim = self.head_dim // 2
        real = real.reshape(value.shape[0], value.shape[1], self.heads, complex_head_dim)
        imag = imag.reshape(value.shape[0], value.shape[1], self.heads, complex_head_dim)
        return torch.cat((real, imag), dim=-1).transpose(1, 2)

    def _merge_temporal_heads(self, value: torch.Tensor) -> torch.Tensor:
        value = value.transpose(1, 2)
        real, imag = value.chunk(2, dim=-1)
        real = real.reshape(value.shape[0], value.shape[1], self.hidden // 2)
        imag = imag.reshape(value.shape[0], value.shape[1], self.hidden // 2)
        return torch.cat((real, imag), dim=-1)

    def _project_temporal_kv(
        self,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.temporal_k(self.memory_norm(memory))
        value = self.temporal_v(memory)
        return key, value

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

        def read(
            read_memory: torch.Tensor,
            read_mode_key: torch.Tensor,
            read_times: torch.Tensor,
            read_value_mask: torch.Tensor,
            read_query: torch.Tensor,
        ) -> torch.Tensor:
            key, value = self._project_temporal_kv(read_memory)

            key = key + read_mode_key[None]

            value = value * read_value_mask.to(value.dtype)[None, :, None]
            key = self._temporal_heads(key)
            value = self._temporal_heads(value)
            key = _time_rope(key, read_times, rope_dim=self.time_rope_dim)
            return F.scaled_dot_product_attention(read_query, key, value)

        current_memory = flat_memory.index_select(1, current_memory_indices)
        current_key = mode_key.index_select(0, current_memory_indices)
        current_times = expanded_memory.index_select(1, current_memory_indices)
        current_value_mask = value_mask.index_select(0, current_memory_indices)
        current_read = read(
            current_memory,
            current_key,
            current_times,
            current_value_mask,
            q,
        )
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
        prediction = query + float(action_scale) * innovation
        return prediction


class FourierWorld(nn.Module):
    """Predict a future patch grid from four uniformly spaced feature grids."""

    supports_time_step = True
    basis_name = "fourier"
    method_revision = "fourierworld_learned_mask_query"

    def __init__(
        self,
        *,
        feature_dim: int = 768,
        hidden: int = 768,
        depth: int = 9,
        heads: int = 12,
        intermediate: int = 3376,
        spectral_eps: float = 1e-4,
        reference_dt: float = 0.2,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        if feature_dim != hidden:
            raise ValueError("parameter-free dense carrier requires feature_dim == hidden")
        if hidden % 2 or hidden % heads or (hidden // heads) % 4:
            raise ValueError("hidden must support the complex lift and 2-D RoPE")
        if min(depth, intermediate) < 1 or spectral_eps <= 0 or reference_dt <= 0:
            raise ValueError("model dimensions and physical scales must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.heads = int(heads)
        self.intermediate = int(intermediate)
        self.spectral_eps = float(spectral_eps)
        self.reference_dt = float(reference_dt)

        self.dc_lift = nn.Linear(feature_dim, hidden, bias=False)
        self.ac_lift = ComplexModeLift(feature_dim, hidden)
        self.nyquist_lift = nn.Linear(feature_dim, hidden, bias=False)

        self.mode_key = nn.Parameter(torch.empty(3, hidden))
        nn.init.trunc_normal_(self.mode_key, std=0.02)
        self.blocks = nn.ModuleList(
            FourierStateQueryBlock(hidden, heads, intermediate, layer_scale_init=layer_scale_init)
            for _ in range(depth)
        )
        self.query_mask = nn.Parameter(torch.empty(1, 1, hidden))
        nn.init.normal_(self.query_mask, std=0.02)

    @property
    def active_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _default_times(self, value: torch.Tensor) -> torch.Tensor:
        return self.reference_dt * torch.arange(
            value.shape[1], device=value.device, dtype=torch.float32
        )[None].expand(value.shape[0], -1)

    @staticmethod
    def _resolve_spatial_shape(
        tokens: int,
        spatial_shape: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if spatial_shape is None:
            side = math.isqrt(tokens)
            if side * side != tokens:
                raise ValueError("non-square token grids require spatial_shape")
            return side, side
        height, width = map(int, spatial_shape)
        if height <= 0 or width <= 0 or height * width != tokens:
            raise ValueError("spatial_shape does not match token count")
        return height, width

    def _validate_context(self, context: torch.Tensor) -> None:
        if context.ndim != 4 or context.shape[1] < 4 or context.shape[-1] != self.feature_dim:
            raise ValueError(f"expected [B,T>=4,S,{self.feature_dim}], got {tuple(context.shape)}")

    def analyze(
        self,
        context: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_context(context)
        windows = torch.stack(
            [context[:, start : start + 4] for start in range(context.shape[1] - 3)],
            dim=1,
        )
        spectrum = torch.fft.rfft(windows.float(), n=4, dim=2, norm="ortho")
        scale = windows.float().square().mean(dim=(2, 4)).add(self.spectral_eps).sqrt()[..., None]
        return spectrum, scale, timestamps[:, 3:]

    def _memory_tokens(
        self,
        spectrum: torch.Tensor,
        scale: torch.Tensor,
        window_times: torch.Tensor,
        *,
        dc_scale: float,
        ac_scale: float,
        imaginary_scale: float,
        nyquist_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dc = spectrum[:, :, 0].real / scale
        ac = spectrum[:, :, 1]
        ac_real, ac_imag = ac.real / scale, ac.imag / scale
        nyquist = spectrum[:, :, 2].real / scale

        raw_motion = torch.stack((ac_real, ac_imag, nyquist), dim=2)
        motion_mask = (raw_motion.abs().amax(dim=(1, 2, 4), keepdim=False) > 0)[..., None]

        dc_token = float(dc_scale) * self.dc_lift(dc)
        ac_token = self.ac_lift(
            float(ac_scale) * ac_real,
            float(ac_scale) * float(imaginary_scale) * ac_imag,
        )
        nyquist_token = float(nyquist_scale) * self.nyquist_lift(nyquist)

        tokens = torch.stack((dc_token[:, -1], ac_token[:, -1], nyquist_token[:, -1]), dim=2)
        windows = spectrum.shape[1]
        modes = torch.arange(3, device=tokens.device)
        window_indices = torch.full((3,), windows - 1, device=tokens.device, dtype=torch.long)
        return tokens, modes, window_indices, motion_mask

    def _predict_from_analysis(
        self,
        latest: torch.Tensor,
        spectrum: torch.Tensor,
        scale: torch.Tensor,
        window_times: torch.Tensor,
        future_times: torch.Tensor,
        *,
        spatial_shape: tuple[int, int],
        spatial_coordinates: torch.Tensor | None = None,
        dc_scale: float = 1.0,
        ac_scale: float = 1.0,
        imaginary_scale: float = 1.0,
        nyquist_scale: float = 1.0,
        action_scale: float = 1.0,
    ) -> torch.Tensor:
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
        mode_key = self.mode_key.index_select(0, modes)
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

    def forward(
        self,
        context: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        *,
        future_timestamp: float | torch.Tensor | None = None,
        spatial_shape: tuple[int, int] | None = None,
        spatial_coordinates: torch.Tensor | None = None,
        dc_scale: float = 1.0,
        ac_scale: float = 1.0,
        imaginary_scale: float = 1.0,
        nyquist_scale: float = 1.0,
        action_scale: float = 1.0,
    ) -> torch.Tensor:
        self._validate_context(context)
        batch, observed_steps, spatial, _ = context.shape
        if timestamps is None:
            context = context[:, -4:]
            times = self._default_times(context)
        else:
            times = _validate_times(timestamps, batch, observed_steps).to(context.device)[:, -4:]
            context = context[:, -4:]
        if future_timestamp is None:
            future = times[:, -1:] + (times[:, -1:] - times[:, -2:-1])
        elif isinstance(future_timestamp, Real):
            future = torch.full((batch, 1), float(future_timestamp), device=context.device)
        elif torch.is_tensor(future_timestamp):
            future = future_timestamp.to(context.device, torch.float32)
            if future.ndim == 0:
                future = future.expand(batch, 1)
            elif future.shape == (batch,):
                future = future[:, None]
        else:
            raise TypeError("future_timestamp must be real or tensor")
        if future.shape != (batch, 1) or not (future > times[:, -1:]).all():
            raise ValueError("future timestamp must be [B,1] and after the context")
        spectrum, scale, window_times = self.analyze(context, times)
        shape = self._resolve_spatial_shape(spatial, spatial_shape)
        return self._predict_from_analysis(
            context[:, -1],
            spectrum,
            scale,
            window_times,
            future,
            spatial_shape=shape,
            spatial_coordinates=spatial_coordinates,
            dc_scale=dc_scale,
            ac_scale=ac_scale,
            imaginary_scale=imaginary_scale,
            nyquist_scale=nyquist_scale,
            action_scale=action_scale,
        )

    def initialize(
        self,
        context: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> FourierQueryState:
        self._validate_context(context)
        times = (
            self._default_times(context)
            if timestamps is None
            else _validate_times(timestamps, context.shape[0], context.shape[1]).to(context.device)
        )
        context = context[:, -4:]
        times = times[:, -4:]
        spectrum, scale, window_times = self.analyze(context, times)
        return {
            "tail": context[:, -3:],
            "tail_times": times[:, -3:],
            "spectrum": spectrum,
            "window_scale": scale,
            "window_times": window_times,
        }

    def step(
        self,
        state: FourierQueryState,
        *,
        future_timestamp: float | torch.Tensor | None = None,
        spatial_shape: tuple[int, int] | None = None,
        spatial_coordinates: torch.Tensor | None = None,
        **interventions: float,
    ) -> tuple[torch.Tensor, FourierQueryState]:
        tail, tail_times = state["tail"], state["tail_times"]
        batch, _, spatial, _ = tail.shape
        if future_timestamp is None:
            future = tail_times[:, -1:] + (tail_times[:, -1:] - tail_times[:, -2:-1])
        elif isinstance(future_timestamp, Real):
            future = torch.full((batch, 1), float(future_timestamp), device=tail.device)
        else:
            future = future_timestamp.to(tail.device, torch.float32)
            if future.ndim == 0:
                future = future.expand(batch, 1)
            elif future.shape == (batch,):
                future = future[:, None]
        if future.shape != (batch, 1) or not (future > tail_times[:, -1:]).all():
            raise ValueError("future timestamp must follow the recurrent tail")
        shape = self._resolve_spatial_shape(spatial, spatial_shape)
        prediction = self._predict_from_analysis(
            tail[:, -1],
            state["spectrum"],
            state["window_scale"],
            state["window_times"],
            future,
            spatial_shape=shape,
            spatial_coordinates=spatial_coordinates,
            **interventions,
        )
        new_window = torch.cat((tail, prediction[:, None]), dim=1)
        new_times = torch.cat((tail_times, future), dim=1)

        new_spectrum, new_scale, _ = self.analyze(new_window, new_times)
        next_state: FourierQueryState = {
            "tail": new_window[:, 1:],
            "tail_times": new_times[:, 1:],
            "spectrum": new_spectrum,
            "window_scale": new_scale,
            "window_times": future,
        }
        return prediction, next_state

    def rollout(
        self,
        context: torch.Tensor,
        steps: int,
        timestamps: torch.Tensor | None = None,
        *,
        spatial_shape: tuple[int, int] | None = None,
        spatial_coordinates: torch.Tensor | None = None,
        **interventions: float,
    ) -> list[torch.Tensor]:
        if int(steps) < 1:
            raise ValueError("steps must be positive")
        initial_context = context[:, -4:]
        initial_times = timestamps if timestamps is None else timestamps[:, -4:]
        state = self.initialize(initial_context, initial_times)
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
