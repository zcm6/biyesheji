from __future__ import annotations

import numpy as np

def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)) if data else np.zeros(0, dtype=np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    if len(bits) == 0:
        return b""
    pad = (-len(bits)) % 8
    if pad:
        bits = np.pad(bits.astype(np.uint8, copy=False), (0, pad))
    return np.packbits(bits.astype(np.uint8, copy=False)).tobytes()


def ints_to_bits(values: list[int] | np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return np.zeros(0, dtype=np.uint8)
    array = np.asarray(values, dtype=np.uint16)
    if array.size == 0:
        return np.zeros(0, dtype=np.uint8)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint16)
    return ((array[:, None] >> shifts) & 1).astype(np.uint8, copy=False).reshape(-1)


"""
    将一维比特流按固定宽度切分，转换为整数列表。

    转换逻辑遵循“大端模式”，即数组中先出现的比特作为整数的高位。
    如果比特长度不是 width 的整数倍，会在末尾自动补 0。

    Args:
        bits (np.ndarray): 输入的比特流数组（由 0 和 1 组成）。
        width (int): 每个整数包含的比特数（如 QPSK 为 2，16QAM 为 4）。

    Returns:
        list[int]: 转换后的整数列表。

    Example:
        bits = [1, 0, 1, 1], width = 2
        -> 分组: [1, 0] 和 [1, 1]
        -> 转换: 2 (二进制 10) 和 3 (二进制 11)
        -> 返回: [2, 3]
"""
def bits_to_ints(bits: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return np.zeros(0, dtype=np.int32)
    data = np.asarray(bits, dtype=np.uint8)
    if data.size == 0:
        return np.zeros(0, dtype=np.int32)
    pad = (-data.size) % width
    if pad:
        data = np.pad(data, (0, pad))
    weights = (1 << np.arange(width - 1, -1, -1, dtype=np.uint16)).astype(np.uint16)
    blocks = data.reshape(-1, width).astype(np.uint16, copy=False)
    return np.sum(blocks * weights, axis=1, dtype=np.uint32).astype(np.int32, copy=False)


def binary_to_gray(values: np.ndarray | list[int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    if array.size == 0:
        return np.zeros(0, dtype=np.int32)
    gray = np.bitwise_xor(array, array >> 1)
    return gray.astype(np.int32, copy=False)


def gray_to_binary(values: np.ndarray | list[int]) -> np.ndarray:
    gray = np.asarray(values, dtype=np.int32)
    if gray.size == 0:
        return np.zeros(0, dtype=np.int32)
    binary = gray.copy()
    shifted = gray.copy()
    while True:
        shifted = shifted >> 1
        if not np.any(shifted):
            break
        binary = np.bitwise_xor(binary, shifted)
    return binary.astype(np.int32, copy=False)

def grayecode(side: int, dim_width: int, bit_groups: np.ndarray) -> np.ndarray:
    q_bin = bits_to_ints(bit_groups[:, :dim_width].reshape(-1), dim_width)
    i_bin = bits_to_ints(bit_groups[:, dim_width:].reshape(-1), dim_width)
    q_gray = binary_to_gray(q_bin)
    i_gray = binary_to_gray(i_bin)
    indices = (q_gray * side + i_gray).astype(np.int32, copy=False)
    return indices

def graydecode(side: int, dim_width: int, detected_indices : np.ndarray) -> np.ndarray:
    q_gray = detected_indices // side
    i_gray = detected_indices % side
    q_bin = gray_to_binary(q_gray)
    i_bin = gray_to_binary(i_gray)
    q_bits = ints_to_bits(q_bin, dim_width).reshape(-1, dim_width)
    i_bits = ints_to_bits(i_bin, dim_width).reshape(-1, dim_width)
    bits = np.concatenate([q_bits, i_bits], axis=1).reshape(-1).astype(np.int8, copy= False)
    return bits
