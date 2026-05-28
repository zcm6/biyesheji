from __future__ import annotations

import math

import numpy as np

from .constants import SPS


def apply_channel(
    tx_signal: np.ndarray,
    tx_symbols: np.ndarray,
    channel_name: str,
    snr_db: float,
    k_factor: float,
    pulse: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """对发送信号施加信道衰落和加性高斯白噪声。

    Args:
        tx_signal: 调制并脉冲成形后的发送端复基带信号。
        tx_symbols: 调制得到的发送符号序列，用于生成符号级衰落系数。
        channel_name: 信道模型名称，支持 ``"AWGN"``、``"瑞利衰落"`` 和
            ``"莱斯衰落"``。
        snr_db: 信噪比，单位为 dB。
        k_factor: 莱斯衰落信道的 K 因子。
        pulse: 脉冲成形滤波器系数。存在衰落信道时用于重新生成带衰落的
            成形信号。
        rng: 可选的 NumPy 随机数生成器，用于复现实验结果。

    Returns:
        一个二元组，包含经过信道和噪声后的接收信号，以及符号级复衰落系数。
    """
    rng = rng or np.random.default_rng()
    fading = np.ones(len(tx_symbols), dtype=np.complex64)
    if channel_name == "瑞利衰落":
        fading = ((rng.normal(size=len(tx_symbols)) + 1j * rng.normal(size=len(tx_symbols))) / np.sqrt(2)).astype(np.complex64)
    elif channel_name == "莱斯衰落":
        los = math.sqrt(k_factor / (k_factor + 1))
        scatter = math.sqrt(1 / (k_factor + 1))
        fading = (
            los + scatter * (rng.normal(size=len(tx_symbols)) + 1j * rng.normal(size=len(tx_symbols))) / np.sqrt(2)
        ).astype(np.complex64)
    if len(fading):
        fading /= np.sqrt(np.mean(np.abs(fading) ** 2, dtype=np.float64) + 1e-12).astype(np.float32, copy = False)

    if channel_name == "AWGN":
        rx = tx_signal.astype(np.complex64, copy=False)
    elif pulse is not None:
        upsampled = np.zeros(len(tx_symbols) * SPS, dtype=np.complex64)
        upsampled[::SPS] = (tx_symbols * fading).astype(np.complex64, copy=False)
        rx = np.convolve(upsampled, pulse.astype(np.float32), mode="full").astype(np.complex64)
        if len(rx) > len(tx_signal):
            rx = rx[: len(tx_signal)]
        elif len(rx) < len(tx_signal):
            rx = np.r_[rx, np.zeros(len(tx_signal) - len(rx), dtype=np.complex64)]
    else:
        fading_samples = np.repeat(fading, SPS)
        if len(fading_samples) < len(tx_signal):
            fading_samples = np.r_[fading_samples, np.repeat(fading[-1], len(tx_signal) - len(fading_samples))]
        rx = tx_signal * fading_samples[: len(tx_signal)]
    power = np.mean(np.abs(rx) ** 2, dtype=np.float64) + 1e-12
    noise_power = power / (10 ** (snr_db / 10))
    noise = rng.normal(scale=np.sqrt(noise_power / 2), size=len(rx)) + 1j * rng.normal(
        scale=np.sqrt(noise_power / 2), size=len(rx)
    )
    return (rx + noise).astype(np.complex64, copy=False), fading.astype(np.complex64, copy=False)
