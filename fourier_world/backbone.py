"""Frozen DINOv3 ViT-B/16 dense patch feature extraction."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch
from torch import nn
from huggingface_hub import hf_hub_download
from transformers import (
    AutoImageProcessor,
    AutoModel,
    DINOv3ViTConfig,
    DINOv3ViTImageProcessorFast,
    DINOv3ViTModel,
)


OFFICIAL_VITB16_FILENAME = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

_ORIGINAL_TO_HF = (
    (r"cls_token", r"embeddings.cls_token"),
    (r"mask_token", r"embeddings.mask_token"),
    (r"storage_tokens", r"embeddings.register_tokens"),
    (r"patch_embed.proj", r"embeddings.patch_embeddings"),
    (r"periods", r"inv_freq"),
    (r"rope_embed", r"rope_embeddings"),
    (r"blocks.(\d+).attn.proj", r"layer.\1.attention.o_proj"),
    (r"blocks.(\d+).attn.", r"layer.\1.attention."),
    (r"blocks.(\d+).ls(\d+).gamma", r"layer.\1.layer_scale\2.lambda1"),
    (r"blocks.(\d+).mlp.fc1", r"layer.\1.mlp.up_proj"),
    (r"blocks.(\d+).mlp.fc2", r"layer.\1.mlp.down_proj"),
    (r"blocks.(\d+).mlp", r"layer.\1.mlp"),
    (r"blocks.(\d+).norm", r"layer.\1.norm"),
    (r"w1", r"gate_proj"),
    (r"w2", r"up_proj"),
    (r"w3", r"down_proj"),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(path).parts
    )
    if not files:
        raise RuntimeError(f"local DINOv3 snapshot is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix()
        digest.update(f"{relative}\0{item.stat().st_size}\0".encode())
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def huggingface_weights_sha256(path: Path) -> str | None:
    weights = path / "model.safetensors"
    if weights.is_file():
        return file_sha256(weights)
    return None


def converted_key(original: str, *, layer_prefix: str = "") -> str:
    value = original
    for pattern, replacement in _ORIGINAL_TO_HF:
        value = re.sub(pattern, replacement, value)
    if value.startswith("layer."):
        value = f"{layer_prefix}{value}"
    return value


def split_original_qkv(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(state)
    for key in [name for name in result if "qkv" in name]:
        value = result.pop(key)
        if value.shape[0] % 3:
            raise RuntimeError(f"DINOv3 QKV axis is not divisible by three: {key}")
        query, key_tensor, value_tensor = torch.chunk(value, 3, dim=0)
        result[key.replace("qkv", "q_proj")] = query
        result[key.replace("qkv", "k_proj")] = key_tensor
        result[key.replace("qkv", "v_proj")] = value_tensor
    return result


def convert_original_vitb16_state(
    original: dict[str, torch.Tensor],
    *,
    layer_prefix: str = "",
) -> dict[str, torch.Tensor]:
    if not isinstance(original, dict) or not original:
        raise RuntimeError("DINOv3 original checkpoint is not a non-empty state dict")
    state = split_original_qkv(original)
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if (
            "bias_mask" in key
            or "attn.k_proj.bias" in key
            or "local_cls_norm" in key
            or key.startswith("projectors.")
        ):
            continue
        new_key = converted_key(key, layer_prefix=layer_prefix)
        if "inv_freq" in new_key:
            continue
        if "embeddings.mask_token" in new_key:
            tensor = tensor.unsqueeze(1)
        if new_key in converted:
            raise RuntimeError(f"duplicate converted DINOv3 key: {new_key}")
        converted[new_key] = tensor
    return converted


def vitb16_config() -> DINOv3ViTConfig:
    return DINOv3ViTConfig(
        patch_size=16,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        proj_bias=True,
        num_register_tokens=4,
        use_gated_mlp=False,
        hidden_act="gelu",
    )


class DINOv3DenseFeatures(nn.Module):
    """Encode RGB clips into patch tokens with frozen DINOv3 ViT-B/16 weights."""

    def __init__(self, name_or_path: str) -> None:
        super().__init__()
        source = Path(name_or_path).expanduser()
        if source.is_file():
            if source.name != OFFICIAL_VITB16_FILENAME:
                raise RuntimeError(
                    "only the official ViT-B/16 LVD-1689M original checkpoint "
                    f"is accepted, got {source.name!r}"
                )
            original = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
            if set(original) == {"state_dict"} and isinstance(original["state_dict"], dict):
                original = original["state_dict"]
            config = vitb16_config()
            backbone = DINOv3ViTModel(config)
            expected_keys = backbone.state_dict().keys()
            layer_prefix = (
                "model." if any(key.startswith("model.layer.") for key in expected_keys) else ""
            )
            converted = convert_original_vitb16_state(original, layer_prefix=layer_prefix)
            backbone.load_state_dict(converted, strict=True)
            processor = DINOv3ViTImageProcessorFast(do_resize=False, do_center_crop=False)
            self.source_kind = "official_original_pth_in_memory_hf_conversion"
            self.source_path = str(source.resolve())
            self.source_sha256 = file_sha256(source)
            self.weights_sha256 = self.source_sha256
        else:
            if source.exists() and not source.is_dir():
                raise RuntimeError(f"unsupported DINOv3 source: {source}")
            processor = AutoImageProcessor.from_pretrained(
                name_or_path, do_resize=False, do_center_crop=False
            )
            backbone = AutoModel.from_pretrained(name_or_path)
            config = backbone.config
            self.source_kind = (
                "huggingface_local_snapshot" if source.is_dir() else "huggingface_model_id"
            )
            self.source_path = str(source.resolve()) if source.is_dir() else name_or_path
            self.source_sha256 = snapshot_sha256(source) if source.is_dir() else None
            self.weights_sha256 = (
                huggingface_weights_sha256(source)
                if source.is_dir()
                else file_sha256(
                    Path(hf_hub_download(repo_id=name_or_path, filename="model.safetensors"))
                )
            )

        self.processor = processor
        self.backbone = backbone
        self.patch_size = int(config.patch_size)
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.initializer_range = float(config.initializer_range)
        self.num_prefix_tokens = int(config.num_register_tokens) + 1
        if not (
            processor.do_rescale
            and processor.do_normalize
            and float(processor.rescale_factor) == 1.0 / 255.0
            and tuple(map(float, processor.image_mean)) == (0.485, 0.456, 0.406)
            and tuple(map(float, processor.image_std)) == (0.229, 0.224, 0.225)
        ):
            raise RuntimeError("unexpected DINOv3 image normalization contract")

        self.register_buffer(
            "pixel_mean",
            torch.tensor(processor.image_mean).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(processor.image_std).reshape(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(frames.shape)}")
        batch, steps, channels, height, width = frames.shape
        if channels != 3 or height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"invalid DINOv3 frame shape {tuple(frames.shape)} for patch size {self.patch_size}"
            )
        pixels = frames.reshape(batch * steps, channels, height, width).float()
        pixels = (pixels / 255.0 - self.pixel_mean) / self.pixel_std
        tokens = self.backbone(pixels).last_hidden_state[:, self.num_prefix_tokens :]
        return tokens.reshape(batch, steps, -1, self.hidden_size)
