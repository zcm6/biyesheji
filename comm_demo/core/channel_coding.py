from __future__ import annotations

from importlib import import_module
import zlib

import numpy as np

from .bits import bits_to_bytes, ints_to_bits


def hamming74_encode(bits: np.ndarray) -> np.ndarray:
    """对输入比特流进行 (7, 4) 汉明编码。

    Args:
        bits: 待编码的一维比特数组，元素应为 0 或 1。

    Returns:
        uint8 类型的一维数组。每 4 个信息位被编码为 7 位汉明码；
        若输入长度不是 4 的整数倍，会在末尾补 0 后再编码。
    """
    # 检查数据长度是否为4的倍数，如果不够则补0对齐
    if len(bits) % 4:
        bits = np.r_[bits, np.zeros((-len(bits)) % 4, dtype=np.uint8)]
    blocks = bits.reshape(-1, 4).astype(np.uint8)    # 将比特流转换为多行4列的矩阵
    d1 = blocks[:, 0]
    d2 = blocks[:, 1]
    d3 = blocks[:, 2]
    d4 = blocks[:, 3]
    encoded = np.empty((len(blocks), 7), dtype=np.uint8)
    encoded[:, 0] = d1 ^ d2 ^ d4
    encoded[:, 1] = d1 ^ d3 ^ d4
    encoded[:, 2] = d1
    encoded[:, 3] = d2 ^ d3 ^ d4
    encoded[:, 4] = d2
    encoded[:, 5] = d3
    encoded[:, 6] = d4
    return encoded.reshape(-1)  # 重新变为一维比特流


def hamming74_decode(bits: np.ndarray, original_len: int) -> np.ndarray:
    """对 (7, 4) 汉明编码比特流进行译码和单比特纠错。

    Args:
        bits: 接收到的汉明编码比特流，可能包含信道误码。
        original_len: 编码前的原始比特长度，用于去除编码补零。

    Returns:
        uint8 类型的一维数组，包含纠错并截断后的原始信息比特。
    """
    if len(bits) % 7:
        bits = np.r_[bits, np.zeros((-len(bits)) % 7, dtype=np.uint8)]
    code = bits.reshape(-1, 7).astype(np.uint8).copy()
    # 计算伴随式
    s1 = code[:, 0] ^ code[:, 2] ^ code[:, 4] ^ code[:, 6]
    s2 = code[:, 1] ^ code[:, 2] ^ code[:, 5] ^ code[:, 6]
    s3 = code[:, 3] ^ code[:, 4] ^ code[:, 5] ^ code[:, 6]
    """这里设计了一个典型排列的汉明码，使得伴随式的十进制数刚好就是错误位的列数"""
    syndrome = s1 + (s2 << 1) + (s3 << 2)  # s1权重为1，s2权重为2，s3权重为4
    error_rows = np.nonzero((syndrome >= 1) & (syndrome <= 7))[0]
    if len(error_rows):
        code[error_rows, syndrome[error_rows] - 1] ^= 1
    decoded = code[:, [2, 4, 5, 6]].reshape(-1)
    return decoded[:original_len].astype(np.uint8, copy=False)


def convolutional_encode(bits: np.ndarray) -> np.ndarray:
    """使用约束长度为 3 的卷积码对比特流进行编码。

    Args:
        bits: 待编码的一维比特数组，元素应为 0 或 1。

    Returns:
        uint8 类型的一维数组，包含速率为 1/2 的卷积编码输出。
    """
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    u = np.concatenate([data, np.zeros(2, dtype=np.uint8)])
    prev1 = np.concatenate([np.zeros(1, dtype=np.uint8), u[:-1]])
    prev2 = np.concatenate([np.zeros(2, dtype=np.uint8), u[:-2]])
    out1 = np.bitwise_xor(np.bitwise_xor(u, prev1), prev2)
    out2 = np.bitwise_xor(u, prev2)
    out = np.empty(u.size * 2, dtype=np.uint8)
    out[0::2] = out1
    out[1::2] = out2
    return out


def viterbi_decode(bits: np.ndarray, original_len: int) -> np.ndarray:
    """使用 Viterbi 算法对卷积编码比特流进行硬判决译码。

    Args:
        bits: 待译码的卷积编码比特流。
        original_len: 卷积编码前的原始比特长度，用于截断尾比特和补零。

    Returns:
        uint8 类型的一维数组，包含恢复后的原始信息比特。
    """
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size % 2:
        data = data[:-1]
    steps = data.size // 2
    if steps == 0:
        return np.zeros(0, dtype=np.uint8)

    symbols = data.reshape(-1, 2)
    prev_state = np.empty((steps, 4), dtype=np.int8)
    inf = 1 << 30
    m0, m1, m2, m3 = 0, inf, inf, inf

    for step in range(steps):
        b0 = int(symbols[step, 0])
        b1 = int(symbols[step, 1])
        d00 = b0 + b1
        d11 = (1 - b0) + (1 - b1)
        d10 = (1 - b0) + b1
        d01 = b0 + (1 - b1)

        c00 = m0 + d00
        c10 = m1 + d11
        if c00 <= c10:
            n0, p0 = c00, 0
        else:
            n0, p0 = c10, 1

        c01 = m2 + d10
        c11 = m3 + d01
        if c01 <= c11:
            n1, p1 = c01, 2
        else:
            n1, p1 = c11, 3

        c02 = m0 + d11
        c12 = m1 + d00
        if c02 <= c12:
            n2, p2 = c02, 0
        else:
            n2, p2 = c12, 1

        c03 = m2 + d01
        c13 = m3 + d10
        if c03 <= c13:
            n3, p3 = c03, 2
        else:
            n3, p3 = c13, 3

        m0, m1, m2, m3 = n0, n1, n2, n3
        prev_state[step, 0] = p0
        prev_state[step, 1] = p1
        prev_state[step, 2] = p2
        prev_state[step, 3] = p3

    state = 0
    best_metric = m0
    if m1 < best_metric:
        state, best_metric = 1, m1
    if m2 < best_metric:
        state, best_metric = 2, m2
    if m3 < best_metric:
        state = 3

    decoded = np.empty(steps, dtype=np.uint8)
    for step in range(steps - 1, -1, -1):
        decoded[step] = (state >> 1) & 1
        state = int(prev_state[step, state])
    return decoded[:original_len]


def channel_encode(bits: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    """对信源编码比特流进行信道编码或 CRC 校验附加。

    Args:
        bits: 来自信源编码模块的一维比特数组。
        method: 信道编码方式。``"CRC"`` 表示附加 8 位 CRC 校验，
            ``"汉明码"`` 表示使用 (7, 4) 汉明码，其他取值默认使用卷积码。

    Returns:
        一个二元组，包含编码后的信道比特流，以及记录原始长度的元数据字典。
    """
    if method == "CRC":
        crc = zlib.crc32(bits_to_bytes(bits)) & 0xFF   # 将32位校验码只保留最后8位
        return np.r_[bits, ints_to_bits([crc], 8)], {"length": len(bits)}  # 将数据与校验数据进行拼接
    if method == "汉明码":
        return hamming74_encode(bits), {"length": len(bits)}
    return convolutional_encode(bits), {"length": len(bits)}


def _ai_convolutional_decode(
    bits: np.ndarray,
    original_len: int,
    modulation: str,
    order: int,
    channel_name: str,
) -> np.ndarray:
    """调用外部 AI 模型对卷积码比特流进行译码。

    Args:
        bits: 待译码的卷积编码比特流。
        original_len: 卷积编码前的原始比特长度。
        modulation: 当前链路使用的调制方式。
        order: 当前链路使用的调制阶数。
        channel_name: 当前链路使用的信道模型名称。

    Returns:
        AI 译码器恢复出的原始信息比特数组。
    """
    inference = import_module("ai卷积码译码.inference")
    return inference.bigru_decode(
        bits,
        original_len,
        modulation,
        order,
        input_mode="hard",
        channel_name=channel_name,
    )


def channel_decode(
    bits: np.ndarray,
    meta: dict,
    method: str,
    *,
    ai_decoder: bool = False,
    modulation: str | None = None,
    order: int | None = None,
    channel_name: str = "AWGN",
) -> tuple[np.ndarray, bool | None]:
    """对接收端比特流进行信道解码或 CRC 校验。

    Args:
        bits: 解调后得到的信道编码比特流。
        meta: 信道编码阶段保存的元数据，需包含原始比特长度 ``length``。
        method: 信道编码方式，需要与编码阶段保持一致。
        ai_decoder: 是否使用 AI 卷积码译码器。
        modulation: AI 译码时使用的调制方式。
        order: AI 译码时使用的调制阶数。
        channel_name: AI 译码时使用的信道模型名称。

    Returns:
        一个二元组，包含恢复后的信源编码比特流，以及 CRC 校验结果。
        非 CRC 编码方式下第二项返回 ``None``。

    Raises:
        ValueError: 当请求 AI 译码非卷积码，或 AI 译码缺少调制参数时抛出。
    """
    original_len = meta["length"]
    if ai_decoder and method != "卷积码":
        raise ValueError("AI 译码当前仅支持卷积码。")
    if method == "CRC":
        payload = bits[:-8] if len(bits) >= 8 else np.zeros(0, dtype=np.uint8)
        recv_crc = bits_to_bytes(bits[-8:])[0] if len(bits) >= 8 else 0
        calc_crc = zlib.crc32(bits_to_bytes(payload)) & 0xFF
        return payload[:original_len], recv_crc == calc_crc
    if method == "汉明码":
        return hamming74_decode(bits, original_len), None
    if ai_decoder:
        if modulation is None or order is None:
            raise ValueError("AI 卷积码译码需要调制方式和调制阶数。")
        return _ai_convolutional_decode(bits, original_len, modulation, order, channel_name), None
    return viterbi_decode(bits, original_len), None
