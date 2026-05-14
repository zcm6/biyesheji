from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from heapq import heappop, heappush

import numpy as np

class ArithmeticCoder:
    STATE_BITS = 32
    FULL = 1 << STATE_BITS
    HALF = FULL >> 1
    QUARTER = HALF >> 1
    THREE_QUARTER = QUARTER * 3
    MASK = FULL - 1
    EOF_SYMBOL = 256

    @staticmethod
    def build_frequencies(data: bytes) -> list[int]:
        freq = [1] * 257
        for value in data:
            freq[value] += 1
        return freq

    @staticmethod
    def cumulative(freq: list[int]) -> list[int]:
        total = 0
        cum = [0]
        for item in freq:
            total += item
            cum.append(total)
        return cum

    @classmethod
    def encode(cls, data: bytes) -> tuple[np.ndarray, dict]:
        freq = cls.build_frequencies(data)
        cum = cls.cumulative(freq)
        total = cum[-1]
        low, high, pending = 0, cls.MASK, 0
        bits: list[int] = []

        def emit(bit: int) -> None:
            nonlocal pending
            bits.append(bit)
            opposite = 1 - bit
            bits.extend([opposite] * pending)
            pending = 0

        for symbol in list(data) + [cls.EOF_SYMBOL]:
            rng = high - low + 1
            high = low + (rng * cum[symbol + 1] // total) - 1
            low = low + (rng * cum[symbol] // total)
            while True:
                if high < cls.HALF:
                    emit(0)
                elif low >= cls.HALF:
                    emit(1)
                    low -= cls.HALF
                    high -= cls.HALF
                elif low >= cls.QUARTER and high < cls.THREE_QUARTER:
                    pending += 1
                    low -= cls.QUARTER
                    high -= cls.QUARTER
                else:
                    break
                low = (low << 1) & cls.MASK
                high = ((high << 1) & cls.MASK) | 1
        pending += 1
        emit(0 if low < cls.QUARTER else 1)
        return np.array(bits, dtype=np.uint8), {"freq": freq}

    @classmethod
    def decode(cls, bits: np.ndarray, meta: dict) -> bytes:
        freq = meta["freq"]
        cum = cls.cumulative(freq)
        total = cum[-1]
        stream = bits.tolist() + [0] * (cls.STATE_BITS + len(bits) + 1024)
        low, high, code = 0, cls.MASK, 0
        pos = 0
        for _ in range(cls.STATE_BITS):
            code = ((code << 1) | stream[pos]) & cls.MASK
            pos += 1
        out = bytearray()
        while True:
            rng = high - low + 1
            value = ((code - low + 1) * total - 1) // rng
            symbol = bisect_right(cum, value) - 1
            if symbol == cls.EOF_SYMBOL:
                break
            out.append(symbol)
            high = low + (rng * cum[symbol + 1] // total) - 1
            low = low + (rng * cum[symbol] // total)
            while True:
                if high < cls.HALF:
                    pass
                elif low >= cls.HALF:
                    low -= cls.HALF
                    high -= cls.HALF
                    code -= cls.HALF
                elif low >= cls.QUARTER and high < cls.THREE_QUARTER:
                    low -= cls.QUARTER
                    high -= cls.QUARTER
                    code -= cls.QUARTER
                else:
                    break
                low = (low << 1) & cls.MASK
                high = ((high << 1) & cls.MASK) | 1
                next_bit = stream[pos] if pos < len(stream) else 0
                code = ((code << 1) & cls.MASK) | next_bit
                pos += 1
        return bytes(out)


    """
    将字节数据转换为一维二进制比特数组（0和1）。

    该函数将输入的字节流按位拆解，例如将一个字节 0xFF 拆解为 [1,1,1,1,1,1,1,1]。
    如果输入数据为空，则返回一个空数组。

    Args:
        data (bytes): 输入的字节流数据。

    Returns:
        np.ndarray: 由 0 和 1 组成的 NumPy 数组，数据类型为 uint8。
                    数组长度为输入字节数的 8 倍。
    """

def build_huffman_codes(data: bytes) -> dict[int, str]:
    heap: list[tuple[int, int, object]] = []
    index = 0
    for symbol, weight in Counter(data).items():
        heappush(heap, (weight, index, symbol))
        index += 1
    if len(heap) == 1:
        return {heap[0][2]: "0"}
    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)
        heappush(heap, (left[0] + right[0], index, (left, right)))
        index += 1
    codes: dict[int, str] = {}

    def walk(node: object, prefix: str) -> None:
        if isinstance(node, int):
            codes[node] = prefix or "0"
            return
        walk(node[0][2], prefix + "0")
        walk(node[1][2], prefix + "1")

    walk(heap[0][2], "")
    return codes


def build_shannon_fano_codes(data: bytes) -> dict[int, str]:
    items = sorted(Counter(data).items(), key=lambda item: (-item[1], item[0]))
    codes = {symbol: "" for symbol, _ in items}

    def split(block: list[tuple[int, int]]) -> None:
        if len(block) <= 1:
            return
        total = sum(freq for _, freq in block)
        acc = 0
        cut = 1
        best_gap = total
        for idx in range(1, len(block)):
            acc += block[idx - 1][1]
            gap = abs(total - 2 * acc)
            if gap < best_gap:
                best_gap = gap
                cut = idx
        left, right = block[:cut], block[cut:]
        for symbol, _ in left:
            codes[symbol] += "0"
        for symbol, _ in right:
            codes[symbol] += "1"
        split(left)
        split(right)

    split(items)
    return {symbol: (code or "0") for symbol, code in codes.items()}

"""
    对原始数据进行信源编码（压缩），将其转换为二进制比特流。

    支持三种编码方式：算术编码、哈夫曼编码、香农-范诺编码。
    编码后的数据体积通常小于原始数据（除非数据完全随机）。

    Args:
        data (bytes): 原始的字节流数据（待压缩）。
        method (str): 编码方式，可选值为 "算术编码"、"哈夫曼编码" 或其他（默认为香农-范诺）。

    Returns:
        tuple[np.ndarray, dict]: 包含两个元素的元组：
            - np.ndarray: 编码后的比特流数组（由 0 和 1 组成的 uint8 数组）。
            - dict: 编码元数据字典。
                - 包含 'method' (方法名)。
                - 包含 'length' (原始数据长度，用于解压时校验)。
                - 若为哈夫曼/香农-范诺，包含 'codes' (解码所需的码表)。
"""
def source_encode(data: bytes, method: str) -> tuple[np.ndarray, dict]:
    if method == "算术编码":
        bits, meta = ArithmeticCoder.encode(data)
        meta["method"] = method   # 记录使用的方法名
        return bits, meta
    codes = build_huffman_codes(data) if method == "哈夫曼编码" else build_shannon_fano_codes(data)
    bit_string = "".join(codes[byte] for byte in data)
    if bit_string:
        bits = (np.frombuffer(bit_string.encode("ascii"), dtype=np.uint8) - ord("0")).astype(np.uint8, copy=False)
    else:
        bits = np.zeros(0, dtype=np.uint8)
    return bits, {"method": method, "codes": codes, "length": len(data)}


def source_decode(bits: np.ndarray, meta: dict, method: str) -> bytes:
    if method == "算术编码":
        return ArithmeticCoder.decode(bits, meta)
    # 将码表转换为二叉前缀树，避免逐 bit 字符串拼接和哈希查找。
    codes = meta["codes"]
    left: list[int] = [-1]
    right: list[int] = [-1]
    symbol_at: list[int] = [-1]
    for symbol, code in codes.items():
        node = 0
        for token in code:
            if token == "0":
                nxt = left[node]
                if nxt < 0:
                    nxt = len(left)
                    left[node] = nxt
                    left.append(-1)
                    right.append(-1)
                    symbol_at.append(-1)
                node = nxt
            else:
                nxt = right[node]
                if nxt < 0:
                    nxt = len(left)
                    right[node] = nxt
                    left.append(-1)
                    right.append(-1)
                    symbol_at.append(-1)
                node = nxt
        symbol_at[node] = int(symbol)

    bit_stream = np.asarray(bits, dtype=np.uint8).reshape(-1)
    node = 0
    out = bytearray()
    for bit in bit_stream:
        node = left[node] if int(bit) == 0 else right[node]
        if node < 0:
            break
        symbol = symbol_at[node]
        if symbol >= 0:
            out.append(symbol)
            node = 0
            if len(out) == meta["length"]:
                break
    return bytes(out)
