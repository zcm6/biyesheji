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
    """Convert coded bits or soft values to [steps, 2] RNN observations."""
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
    """Run the project's modulation, channel, and demodulation chain."""
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
    """Generate one training batch using the project's full core link."""
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
