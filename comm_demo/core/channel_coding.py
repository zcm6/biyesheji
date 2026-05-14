from __future__ import annotations

import zlib

import numpy as np

from .bits import bits_to_bytes, ints_to_bits

def hamming74_encode(bits: np.ndarray) -> np.ndarray:
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


"""
    对 (7, 4) 汉明编码的比特流进行解码，包含纠错步骤。

    解码流程：
    1. 填充并对齐数据，确保长度是 7 的倍数。
    2. 计算伴随式，检测是否有错以及错误位置。
    3. 根据伴随式纠正错误位（翻转错误的比特）。
    4. 提取有效的数据位，丢弃校验位。
    5. 根据原始长度截断数据，去除填充的 0。

    Args:
        bits (np.ndarray): 接收到的编码比特流（可能包含噪声错误）。
        original_len (int): 编码前的原始数据长度，用于去除填充位。

    Returns:
        np.ndarray: 解码并纠错后的原始比特流。
"""
def hamming74_decode(bits: np.ndarray, original_len: int) -> np.ndarray:
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

"""
    对比特流进行信道编码，添加冗余信息以实现检错或纠错功能。
    
    支持三种编码方式：
    - "CRC": 循环冗余校验，添加 8 位校验位（仅检错）。
    - "汉明码": (7,4) 汉明码，添加校验位（可纠错）。
    - 其他: 默认为卷积编码（通常用于纠错）。

    Args:
        bits (np.ndarray): 输入的比特流数组（由 0 和 1 组成）。
        method (str): 编码方式，可选 "CRC"、"汉明码" 或其他。

    Returns:
        tuple[np.ndarray, dict]: 包含两个元素的元组：
            - np.ndarray: 编码后的比特流数组（长度通常会增加）。
            - dict: 元数据字典，包含 'length' (原始数据长度)，用于解码时截取有效数据。
"""
def channel_encode(bits: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    if method == "CRC":
        crc = zlib.crc32(bits_to_bytes(bits)) & 0xFF   # 将32位校验码只保留最后8位
        return np.r_[bits, ints_to_bits([crc], 8)], {"length": len(bits)}  # 将数据与校验数据进行拼接
    if method == "汉明码":
        return hamming74_encode(bits), {"length": len(bits)}
    return convolutional_encode(bits), {"length": len(bits)}


def channel_decode(bits: np.ndarray, meta: dict, method: str) -> tuple[np.ndarray, bool | None]:
    original_len = meta["length"]
    if method == "CRC":
        payload = bits[:-8] if len(bits) >= 8 else np.zeros(0, dtype=np.uint8)
        recv_crc = bits_to_bytes(bits[-8:])[0] if len(bits) >= 8 else 0
        calc_crc = zlib.crc32(bits_to_bytes(payload)) & 0xFF
        return payload[:original_len], recv_crc == calc_crc
    if method == "汉明码":
        return hamming74_decode(bits, original_len), None
    return viterbi_decode(bits, original_len), None
