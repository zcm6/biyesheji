from __future__ import annotations

"""AI-BiGRU 卷积码译码器推理工具。

该模块根据调制方式、调制阶数和信道类型选择对应的 AWGN 专用模型，
并将接收端硬判决比特或软信息送入 BiGRU 模型恢复原始信息比特。
"""

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
    """返回当前已配置模型支持的调制组合。

    Returns:
        由 ``(modulation, order)`` 组成的元组列表。
    """
    return tuple(MODEL_FILES_BY_CONFIG)


def select_model_path(modulation: str, order: int, channel_name: str = "AWGN") -> Path:
    """根据调制配置选择对应的 AI-BiGRU 模型路径。

    Args:
        modulation: 当前链路使用的调制方式。
        order: 当前链路使用的调制阶数。
        channel_name: 当前链路使用的信道模型名称。

    Returns:
        与调制配置匹配的模型权重文件路径。

    Raises:
        ValueError: 当信道不是 AWGN，或当前调制配置没有可用模型时抛出。
    """
    if channel_name != "AWGN":
        raise ValueError(f"当前 AI 卷积码专用模型仅支持 AWGN，收到: {channel_name}")
    key = (modulation, int(order))
    if key not in MODEL_FILES_BY_CONFIG:
        raise ValueError(f"未找到 {modulation}-{order} 的 AI 卷积码模型。")
    return MODEL_DIR / MODEL_FILES_BY_CONFIG[key]


def load_model(path: Path, device: str):
    """从检查点文件加载 BiGRU 译码模型。

    Args:
        path: 模型检查点文件路径。
        device: 模型加载到的 PyTorch 设备。

    Returns:
        进入评估模式的 BiGRU 译码模型。
    """
    checkpoint = torch.load(path, map_location="cpu")
    model = ConvCodeRNNDecoder(config_from_dict(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def preferred_device() -> str:
    """选择当前环境优先使用的 PyTorch 推理设备。

    Returns:
        若 CUDA 可用则返回 ``"cuda"``，否则返回 ``"cpu"``。
    """
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
    """使用匹配的 AWGN 专用 BiGRU 模型译码接收序列。

    Args:
        received_values: 接收端硬判决比特或软信息序列。
        original_len: 需要恢复的原始信息比特长度。
        modulation: 当前链路使用的调制方式。
        order: 当前链路使用的调制阶数。
        input_mode: 模型输入模式，支持 ``"hard"`` 和 ``"soft"``。
        channel_name: 当前链路使用的信道模型名称。
        device: 可选的 PyTorch 推理设备；为空时自动选择。
        model_cache: 可选的模型缓存字典，用于避免重复加载同一模型。
        window_steps: 单次送入模型的最大时间步数。

    Returns:
        uint8 类型的一维数组，包含 AI-BiGRU 译码恢复出的信息比特。
    """
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
