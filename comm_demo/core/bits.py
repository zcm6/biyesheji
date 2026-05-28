from __future__ import annotations

import numpy as np


def bytes_to_bits(data: bytes) -> np.ndarray:
    """将字节流转换为一维比特数组。

    Args:
        data: 待转换的字节数据，通常来自信源输入的原始载荷。

    Returns:
        uint8 类型的一维 NumPy 数组。数组元素为 0 或 1，并按字节内
        高位在前的顺序排列；当输入为空时返回空数组。
    """
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)) if data else np.zeros(0, dtype=np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """将一维比特数组按字节打包。

    Args:
        bits: 待打包的比特数组，元素应为 0 或 1。

    Returns:
        打包后的字节数据。若比特数不是 8 的整数倍，会在末尾补 0 后
        再打包；当输入为空时返回空字节串。
    """
    if len(bits) == 0:
        return b""
    pad = (-len(bits)) % 8
    if pad:
        bits = np.pad(bits.astype(np.uint8, copy=False), (0, pad))
    return np.packbits(bits.astype(np.uint8, copy=False)).tobytes()


def ints_to_bits(values: list[int] | np.ndarray, width: int) -> np.ndarray:
    """将整数序列展开为固定宽度的比特流。

    Args:
        values: 待转换的整数序列，例如 CRC 校验值或解调后的星座索引。
        width: 每个整数展开时使用的比特宽度。

    Returns:
        uint8 类型的一维比特数组。每个整数按高位在前的顺序展开；
        当 ``width`` 非正数或输入序列为空时返回空数组。
    """
    if width <= 0:
        return np.zeros(0, dtype=np.uint8)
    array = np.asarray(values, dtype=np.uint16)
    if array.size == 0:
        return np.zeros(0, dtype=np.uint8)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint16)
    return ((array[:, None] >> shifts) & 1).astype(np.uint8, copy=False).reshape(-1)


def bits_to_ints(bits: np.ndarray, width: int) -> np.ndarray:
    """将比特流按固定宽度分组并转换为整数。

    Args:
        bits: 待转换的一维比特数组，元素应为 0 或 1。
        width: 每个整数分组包含的比特数，通常等于调制阶数的
            ``log2(order)``。

    Returns:
        int32 类型的一维整数数组。每组比特按高位在前解释为整数；
        若最后一组不足 ``width`` 位，会在末尾补 0 后再转换。
    """
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
    """将普通二进制整数映射为格雷码整数。

    Args:
        values: 待映射的二进制整数序列，通常是调制符号索引。

    Returns:
        int32 类型的一维数组，包含每个输入整数对应的格雷码值；
        当输入为空时返回空数组。
    """
    array = np.asarray(values, dtype=np.int32)
    if array.size == 0:
        return np.zeros(0, dtype=np.int32)
    gray = np.bitwise_xor(array, array >> 1)
    return gray.astype(np.int32, copy=False)


def gray_to_binary(values: np.ndarray | list[int]) -> np.ndarray:
    """将格雷码整数还原为普通二进制整数。

    Args:
        values: 待还原的格雷码整数序列，通常来自解调判决后的星座索引。

    Returns:
        int32 类型的一维数组，包含每个格雷码值对应的二进制整数；
        当输入为空时返回空数组。
    """
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
    """将 MQAM 的 Q/I 比特分组编码为格雷码星座索引。

    Args:
        side: MQAM 方形星座图单个坐标轴上的点数。
        dim_width: Q 维或 I 维各自占用的比特宽度。
        bit_groups: 二维比特分组数组。每一行表示一个调制符号，
            前 ``dim_width`` 位为 Q 维比特，剩余位为 I 维比特。

    Returns:
        int32 类型的一维数组，包含按 Q/I 两个维度分别格雷编码后
        合成的星座点索引。
    """
    q_bin = bits_to_ints(bit_groups[:, :dim_width].reshape(-1), dim_width)
    i_bin = bits_to_ints(bit_groups[:, dim_width:].reshape(-1), dim_width)
    q_gray = binary_to_gray(q_bin)
    i_gray = binary_to_gray(i_bin)
    indices = (q_gray * side + i_gray).astype(np.int32, copy=False)
    return indices


def graydecode(side: int, dim_width: int, detected_indices : np.ndarray) -> np.ndarray:
    """将 MQAM 格雷码星座索引解码为 Q/I 拼接比特流。

    Args:
        side: MQAM 方形星座图单个坐标轴上的点数。
        dim_width: Q 维或 I 维各自占用的比特宽度。
        detected_indices: 解调判决得到的格雷码星座点索引。

    Returns:
        int8 类型的一维比特数组。每个符号先输出 Q 维比特，
        再输出 I 维比特，并保持与调制端一致的拼接顺序。
    """
    q_gray = detected_indices // side
    i_gray = detected_indices % side
    q_bin = gray_to_binary(q_gray)
    i_bin = gray_to_binary(i_gray)
    q_bits = ints_to_bits(q_bin, dim_width).reshape(-1, dim_width)
    i_bits = ints_to_bits(i_bin, dim_width).reshape(-1, dim_width)
    bits = np.concatenate([q_bits, i_bits], axis=1).reshape(-1).astype(np.int8, copy= False)
    return bits
