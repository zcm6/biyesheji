from __future__ import annotations

import math

import numpy as np

from .bits import bits_to_ints, binary_to_gray, gray_to_binary, graydecode, grayecode, ints_to_bits
from .constants import DEFAULT_SPAN, SPS


def root_raised_cosine(beta: float, span_symbols: int = DEFAULT_SPAN, sps: int = SPS) -> np.ndarray:
    """生成单位能量归一化的根升余弦滤波器。

    Args:
        beta: 根升余弦滤波器的滚降系数。
        span_symbols: 滤波器覆盖的符号跨度。
        sps: 每个符号对应的采样点数。

    Returns:
        float32 类型的一维数组，表示根升余弦脉冲成形滤波器系数。
    """
    t = np.arange(-span_symbols * sps / 2, span_symbols * sps / 2 + 1, dtype=np.float32) / np.float32(sps)
    pulse = np.zeros_like(t, dtype=np.float32)
    for index, value in enumerate(t):
        if abs(value) < 1e-12:
            pulse[index] = 1.0 - beta + 4 * beta / math.pi
        elif beta > 0 and abs(abs(value) - 1 / (4 * beta)) < 1e-12:
            pulse[index] = (
                beta
                / math.sqrt(2)
                * (
                    (1 + 2 / math.pi) * math.sin(math.pi / (4 * beta))
                    + (1 - 2 / math.pi) * math.cos(math.pi / (4 * beta))
                )
            )
        else:
            numerator = math.sin(math.pi * value * (1 - beta)) + 4 * beta * value * math.cos(math.pi * value * (1 + beta))
            denominator = math.pi * value * (1 - (4 * beta * value) ** 2)
            pulse[index] = numerator / denominator
    pulse /= np.sqrt(np.sum(np.abs(pulse) ** 2, dtype=np.float64)).astype(np.float32)
    return pulse.astype(np.float32)


def _scramble_mask(length: int) -> np.ndarray:
    """生成用于比特白化的确定性伪随机掩码。

    Args:
        length: 需要生成的掩码长度。

    Returns:
        uint8 类型的一维数组，元素为 0 或 1；当长度非正时返回空数组。
    """
    if length <= 0:
        return np.zeros(0, dtype=np.uint8)
    # 用于数据白化的确定性伪随机二进制序列。
    rng = np.random.default_rng(20240518)
    return rng.integers(0, 2, size=length, dtype=np.uint8)


def constellation(modulation: str, order: int) -> np.ndarray:
    """根据调制方式和阶数生成归一化星座图。

    Args:
        modulation: 调制方式。``"MASK"`` 生成一维幅度星座，
            ``"MPSK"`` 生成相位星座，其他取值按矩形 QAM 生成。
        order: 调制阶数，即星座点数量。

    Returns:
        complex64 类型的一维数组，包含平均能量归一化后的星座点。
    """
    if modulation == "MASK":
        levels = np.linspace(-(order - 1), order - 1, order, dtype=np.float32)
        points = levels.astype(np.complex64)
    elif modulation == "MPSK":
        points = np.exp(1j * 2 * np.pi * np.arange(order, dtype=np.float32) / np.float32(order)).astype(np.complex64)
    else:
        side = int(round(math.sqrt(order)))
        levels = np.arange(-(side - 1), side + 1, 2, dtype=np.float32)
        points = np.array([complex(i, q) for q in levels[::-1] for i in levels], dtype=np.complex64)
    return (points / np.sqrt(np.mean(np.abs(points) ** 2, dtype=np.float64))).astype(np.complex64)


def modulate(
    bits: np.ndarray, modulation: str, order: int, roll_off: float, gray_ok: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将输入比特流调制为复基带发送信号。

    Args:
        bits: 待调制的一维比特数组。
        modulation: 调制方式，例如 ``"MASK"``、``"MPSK"`` 或 ``"MQAM"``。
        order: 调制阶数。
        roll_off: 根升余弦滤波器滚降系数。
        gray_ok: 是否启用格雷映射。

    Returns:
        一个三元组，依次为脉冲成形后的发送信号、映射后的复数星座符号、
        以及根升余弦滤波器系数。
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size:
        bits = np.bitwise_xor(bits, _scramble_mask(len(bits)))
    width = int(round(math.log2(order)))     # 计算每个符号包含几个比特
    """比特对齐"""
    if len(bits) % width:
        bits = np.r_[bits, np.zeros((-len(bits)) % width, dtype=np.uint8)]
    indices = bits_to_ints(bits, width)  # 把比特流变成整数索引
    if gray_ok:
        if modulation == "MQAM":
            side = int(round(math.sqrt(order)))
            dim_width = int(round(math.log2(side)))
            bit_groups = bits.reshape(-1, width)
            indices = grayecode(side, dim_width, bit_groups)
        else:
            indices = binary_to_gray(indices)
    symbols = constellation(modulation, order)[indices].astype(np.complex64)  # np.complex为复数格式
    upsampled = np.zeros(len(symbols) * SPS, dtype=np.complex64)    # 进行上采样
    upsampled[::SPS] = symbols
    pulse = root_raised_cosine(roll_off)
    signal = np.convolve(upsampled, pulse.astype(np.float32), mode="full").astype(np.complex64)
    return signal, symbols.astype(np.complex64), pulse.astype(np.float32)


DETECTION_CHUNK_SIZE = 8192


def _detect_nearest_points(equalized: np.ndarray, points: np.ndarray) -> np.ndarray:
    """对均衡后的采样符号执行最近星座点判决。

    Args:
        equalized: 均衡后的复数采样符号序列。
        points: 标准星座图上的复数星座点。

    Returns:
        int32 类型的一维数组，包含每个采样符号对应的最近星座点索引。
    """
    if len(equalized) == 0:
        return np.zeros(0, dtype=np.int32)
    indices = np.empty(len(equalized), dtype=np.int32)
    pr = np.real(points).astype(np.float32, copy=False)
    pi = np.imag(points).astype(np.float32, copy=False)
    for start in range(0, len(equalized), DETECTION_CHUNK_SIZE):
        block = equalized[start : start + DETECTION_CHUNK_SIZE]
        br = np.real(block).astype(np.float32, copy=False)
        bi = np.imag(block).astype(np.float32, copy=False)
        dist2 = (br[:, None] - pr[None, :]) ** 2 + (bi[:, None] - pi[None, :]) ** 2
        indices[start : start + len(block)] = np.argmin(dist2, axis=1).astype(np.int32, copy=False)
    return indices


def demodulate(
    rx_signal: np.ndarray,
    pulse: np.ndarray,
    fading: np.ndarray,
    modulation: str,
    order: int,
    expected_bits: int,
    gray_ok: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对接收信号进行匹配滤波、判决和比特恢复。

    Args:
        rx_signal: 经过信道后的接收端复基带信号。
        pulse: 发送端使用的根升余弦滤波器系数。
        fading: 符号级复衰落系数，用于接收端均衡。
        modulation: 调制方式，需要与发送端保持一致。
        order: 调制阶数，需要与发送端保持一致。
        expected_bits: 期望恢复的有效比特数。
        gray_ok: 是否按格雷映射方式进行反映射。

    Returns:
        一个四元组，依次为匹配滤波后的信号、均衡后的采样符号、
        判决得到的星座点，以及恢复出的有效比特流。
    """
    matched = np.convolve(rx_signal, pulse[::-1].conjugate(), mode="full").astype(np.complex64)
    sample_start = len(pulse) - 1
    sample_points = sample_start + np.arange(len(fading)) * SPS
    sampled = matched[sample_points]
    eps = 1e-6
    fading_safe = np.where(
        np.abs(fading) < eps,
        eps * np.exp(1j * np.angle(fading + 1e-12)),
        fading,
    )
    equalized = (sampled / fading_safe).astype(np.complex64, copy = False)
    points = constellation(modulation, order)
    detected_indices = _detect_nearest_points(equalized, points)
    width = int(round(math.log2(order)))
    if gray_ok:
        if modulation == "MQAM":
            side = int(round(math.sqrt(order)))
            dim_width = int(round(math.log2(side)))
            bits = graydecode(side, dim_width, detected_indices)
        else:
            bits = ints_to_bits(gray_to_binary(detected_indices), width)
    else:
        bits = ints_to_bits(detected_indices, width)
    bits = bits[:expected_bits]
    if bits.size:
        bits = np.bitwise_xor(bits.astype(np.uint8, copy=False), _scramble_mask(len(bits)))
    return matched, equalized, points[detected_indices], bits
