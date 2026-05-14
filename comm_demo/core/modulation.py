from __future__ import annotations

import math

import numpy as np

from .bits import bits_to_ints, binary_to_gray, gray_to_binary, graydecode, grayecode, ints_to_bits
from .constants import DEFAULT_SPAN, SPS

def root_raised_cosine(beta: float, span_symbols: int = DEFAULT_SPAN, sps: int = SPS) -> np.ndarray:
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
    if length <= 0:
        return np.zeros(0, dtype=np.uint8)
    # 用于数据白化的确定性伪随机二进制序列。
    rng = np.random.default_rng(20240518)
    return rng.integers(0, 2, size=length, dtype=np.uint8)

"""
    根据调制方式和阶数生成标准化的星座图坐标点。

    生成的星座点会经过功率归一化处理，使其平均能量为 1。
    这确保了不同调制方式下的信噪比（SNR）具有可比性。

    Args:
        modulation (str): 调制方式。
            - "MASK": 多进制幅度键控（一维）。
            - "MPSK": 多进制相移键控（二维，圆周分布）。
            - 其他: 默认为矩形 QAM（正交幅度调制，二维，方形分布）。
        order (int): 调制阶数（星座点总数），如 4, 16, 64。

    Returns:
        np.ndarray: 复数数组，表示星座图上所有点的坐标。
"""
def constellation(modulation: str, order: int) -> np.ndarray:
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


"""
    对比特流进行数字基带调制，生成时域传输信号。

    处理流程：
    1. 比特分组与映射：将比特流按 log2(order) 分组，映射为复数星座点符号。
    2. 上采样：在符号之间插入零，提高采样率。
    3. 脉冲成形：使用根升余弦滤波器进行滤波，限制带宽并成形波形。

    Args:
        bits (np.ndarray): 输入的比特流数组（0和1）。
        modulation (str): 调制方式（如 "BPSK", "QPSK", "QAM"）。
        order (int): 调制阶数（如 2, 4, 16, 64）。
        roll_off (float): 根升余弦滤波器的滚降因子 (0 < roll_off <= 1)。

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 包含三个元素的元组：
            - signal (np.ndarray): 最终生成的时域调制信号（复数基带信号）。
            - symbols (np.ndarray): 映射后的复数符号序列。
            - pulse (np.ndarray): 根升余弦滤波器的冲激响应系数。
"""
def modulate(
    bits: np.ndarray, modulation: str, order: int, roll_off: float, gray_ok: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
