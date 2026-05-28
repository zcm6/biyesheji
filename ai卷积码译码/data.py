from __future__ import annotations
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comm_demo.core.channel_coding import convolutional_encode
from comm_demo.core.channel import apply_channel
from comm_demo.core.modulation import (
    _scramble_mask,
    constellation,
    demodulate,
    gray_to_binary,
    graydecode,
    ints_to_bits,
    modulate,
)


def bits_to_observations(bits: np.ndarray, input_mode: str = "hard") -> np.ndarray:
    """将接收比特或软信息整理为 BiGRU 时间序列输入。

    Args:
        bits: 卷积码接收序列。硬判决模式下为 0/1 比特，软判决模式下为
            浮点软信息。
        input_mode: 输入模式。``"hard"`` 会将 0/1 映射为 -1/+1，
            ``"soft"`` 保留软信息数值。

    Returns:
        float32 类型的二维数组，形状为 ``[steps, 2]``。每个时间步包含
        卷积码速率 1/2 对应的两个接收观测值。
    """
    values = np.asarray(bits, dtype=np.float32).reshape(-1)
    if values.size % 2:
        values = values[:-1]
    if values.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    observations = values.reshape(-1, 2)
    if input_mode == "hard":
        observations = observations * 2.0 - 1.0
    return observations.astype(np.float32, copy=False)


def _constellation_bit_labels(modulation: str, order: int, gray_ok: bool) -> np.ndarray:
    """生成星座点索引对应的比特标签。

    Args:
        modulation: 调制方式。
        order: 调制阶数。
        gray_ok: 是否按照格雷映射关系生成标签。

    Returns:
        int 类型二维数组，形状为 ``[order, log2(order)]``，每一行表示
        对应星座点索引的比特标签。
    """
    width = int(round(np.log2(order)))
    indices = np.arange(order, dtype=np.int32)
    if not gray_ok:
        return ints_to_bits(indices, width).reshape(order, width)
    if modulation == "MQAM":
        side = int(round(np.sqrt(order)))
        dim_width = int(round(np.log2(side)))
        return graydecode(side, dim_width, indices).reshape(order, width)
    return ints_to_bits(gray_to_binary(indices), width).reshape(order, width)


def _soft_bits_from_equalized(
    equalized_symbols: np.ndarray,
    expected_bits: int,
    modulation: str,
    order: int,
    gray_ok: bool,
) -> np.ndarray:
    """根据均衡符号到星座点的距离估计软比特信息。

    Args:
        equalized_symbols: 接收端均衡后的复数星座符号序列。
        expected_bits: 期望输出的有效比特数。
        modulation: 调制方式。
        order: 调制阶数。
        gray_ok: 是否使用格雷映射标签。

    Returns:
        float32 类型的一维软比特数组。数值符号表示更接近 0 或 1，
        绝对值表示相对置信度。
    """
    if expected_bits <= 0:
        return np.zeros(0, dtype=np.float32)

    points = constellation(modulation, order)
    labels = _constellation_bit_labels(modulation, order, gray_ok)
    width = labels.shape[1]
    soft = np.empty(len(equalized_symbols) * width, dtype=np.float32)
    scale = np.mean(np.abs(points) ** 2, dtype=np.float64) + 1e-6

    for symbol_index, value in enumerate(equalized_symbols):
        distances = np.abs(value - points) ** 2
        for bit_index in range(width):
            zero_dist = float(np.min(distances[labels[:, bit_index] == 0]))
            one_dist = float(np.min(distances[labels[:, bit_index] == 1]))
            soft[symbol_index * width + bit_index] = np.clip((zero_dist - one_dist) / scale, -8.0, 8.0)

    soft = soft[:expected_bits]
    if soft.size:
        mask = _scramble_mask(soft.size).astype(bool)
        soft[mask] *= -1.0
    return soft


def project_receive(
    coded_bits: np.ndarray,
    snr_db: float,
    input_mode: str = "hard",
    modulation: str = "MQAM",
    order: int = 16,
    channel_name: str = "AWGN",
    k_factor: float = 3.0,
    roll_off: float = 0.35,
    gray_ok: bool = True,
) -> np.ndarray:
    """运行项目主链路以生成模型接收端观测。

    Args:
        coded_bits: 卷积编码后的发送比特流。
        snr_db: 信道信噪比，单位为 dB。
        input_mode: 模型输入模式，支持 ``"hard"`` 和 ``"soft"``。
        modulation: 调制方式。
        order: 调制阶数。
        channel_name: 信道模型名称。
        k_factor: 莱斯衰落信道的 K 因子。
        roll_off: 根升余弦滤波器滚降系数。
        gray_ok: 是否启用格雷映射。

    Returns:
        float32 类型的二维数组，形状为 ``[steps, 2]``，可直接作为
        BiGRU 译码器的输入观测序列。
    """
    tx_signal, tx_symbols, pulse = modulate(coded_bits, modulation, order, roll_off, gray_ok)
    rx_signal, fading_symbols = apply_channel(tx_signal, tx_symbols, channel_name, snr_db, k_factor, pulse)
    _, equalized_symbols, _, detected_bits = demodulate(
        rx_signal,
        pulse,
        fading_symbols,
        modulation,
        order,
        len(coded_bits),
        gray_ok,
    )
    if input_mode == "soft":
        soft_bits = _soft_bits_from_equalized(equalized_symbols, len(coded_bits), modulation, order, gray_ok)
        return bits_to_observations(soft_bits, input_mode="soft")
    return bits_to_observations(detected_bits, input_mode="hard")


def make_numpy_batch(
    batch_size: int,
    information_len: int,
    snr_min: float,
    snr_max: float,
    rng: np.random.Generator,
    input_mode: str = "hard",
    modulation: str = "MQAM",
    order: int = 16,
    channel_name: str = "AWGN",
    k_factor: float = 3.0,
    roll_off: float = 0.35,
    gray_ok: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """使用完整通信链路生成一个 BiGRU 训练批次。

    Args:
        batch_size: 批次中的样本数量。
        information_len: 每条样本的原始信息比特长度。
        snr_min: 随机采样信噪比的下界，单位为 dB。
        snr_max: 随机采样信噪比的上界，单位为 dB。
        rng: NumPy 随机数生成器。
        input_mode: 模型输入模式，支持 ``"hard"`` 和 ``"soft"``。
        modulation: 调制方式。
        order: 调制阶数。
        channel_name: 信道模型名称。
        k_factor: 莱斯衰落信道的 K 因子。
        roll_off: 根升余弦滤波器滚降系数。
        gray_ok: 是否启用格雷映射。

    Returns:
        一个二元组，包含模型输入 ``x`` 和监督标签 ``y``。``x`` 的形状为
        ``[batch_size, information_len + 2, 2]``，``y`` 的形状为
        ``[batch_size, information_len + 2]``。
    """
    messages = rng.integers(0, 2, size=(batch_size, information_len), dtype=np.uint8)
    steps = information_len + 2
    x = np.empty((batch_size, steps, 2), dtype=np.float32)
    y = np.zeros((batch_size, steps), dtype=np.float32)
    y[:, :information_len] = messages.astype(np.float32)

    for row in range(batch_size):
        coded = convolutional_encode(messages[row])
        snr_db = float(rng.uniform(snr_min, snr_max))
        x[row] = project_receive(
            coded,
            snr_db,
            input_mode=input_mode,
            modulation=modulation,
            order=order,
            channel_name=channel_name,
            k_factor=k_factor,
            roll_off=roll_off,
            gray_ok=gray_ok,
        )
    return x, y
