from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .data import bits_to_observations
    from .model import ConvCodeRNNDecoder, config_from_dict, require_torch, torch
except ImportError:
    from data import bits_to_observations
    from model import ConvCodeRNNDecoder, config_from_dict, require_torch, torch


MODEL_FILES_BY_CONFIG = {
    ("MASK", 2): "bigru_mask_2_awgn.pt",
    ("MASK", 4): "bigru_mask_4_awgn.pt",
    ("MASK", 8): "bigru_mask_8_awgn.pt",
    ("MASK", 16): "bigru_mask_16_awgn.pt",
    ("MPSK", 2): "bigru_mpsk_2_awgn.pt",
    ("MPSK", 4): "bigru_mpsk_4_awgn.pt",
    ("MPSK", 8): "bigru_mpsk_8_awgn.pt",
    ("MPSK", 16): "bigru_mpsk_16_awgn.pt",
    ("MQAM", 16): "bigru_mqam_16_awgn.pt",
}

MODEL_DIR = Path(__file__).resolve().parent / "models"
_MODEL_CACHE: dict[tuple[Path, str], object] = {}


def supported_configs() -> tuple[tuple[str, int], ...]:
    return tuple(MODEL_FILES_BY_CONFIG)


def select_model_path(modulation: str, order: int, channel_name: str = "AWGN") -> Path:
    if channel_name != "AWGN":
        raise ValueError(f"当前 AI 卷积码专用模型仅支持 AWGN，收到: {channel_name}")
    key = (modulation, int(order))
    if key not in MODEL_FILES_BY_CONFIG:
        raise ValueError(f"未找到 {modulation}-{order} 的 AI 卷积码模型。")
    return MODEL_DIR / MODEL_FILES_BY_CONFIG[key]


def load_model(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu")
    model = ConvCodeRNNDecoder(config_from_dict(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def preferred_device() -> str:
    require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def bigru_decode(
    received_values: np.ndarray,
    original_len: int,
    modulation: str,
    order: int,
    input_mode: str = "hard",
    channel_name: str = "AWGN",
    device: str | None = None,
    model_cache: dict[tuple[Path, str], object] | None = None,
    window_steps: int = 8192,
) -> np.ndarray:
    """Decode coded hard bits or soft values with the selected AWGN BiGRU model."""
    require_torch()
    device = device or preferred_device()
    path = select_model_path(modulation, order, channel_name)
    cache = model_cache if model_cache is not None else _MODEL_CACHE
    cache_key = (path, device)
    if cache_key not in cache:
        cache[cache_key] = load_model(path, device)
    observations = bits_to_observations(received_values, input_mode=input_mode)
    decoded = np.empty(min(original_len, len(observations)), dtype=np.uint8)
    if decoded.size == 0:
        return decoded
    with torch.inference_mode():
        for start in range(0, decoded.size, window_steps):
            output_stop = min(start + window_steps, decoded.size)
            observation_stop = min(output_stop + 2, len(observations))
            x = torch.from_numpy(observations[start:observation_stop][None, :, :]).to(device)
            logits = cache[cache_key](x)[0, : output_stop - start]
            decoded[start:output_stop] = (torch.sigmoid(logits) >= 0.5).to(torch.uint8).cpu().numpy()
    return decoded.astype(np.uint8, copy=False)
