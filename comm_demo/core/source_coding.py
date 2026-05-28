from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from heapq import heappop, heappush

import numpy as np


class ArithmeticCoder:
    """基于整数区间更新的算术编码器。

    该类用于将信源字节序列编码为二进制比特流，并在解码阶段根据
    编码时保存的频率表恢复原始字节数据。
    """

    STATE_BITS = 32
    FULL = 1 << STATE_BITS
    HALF = FULL >> 1
    QUARTER = HALF >> 1
    THREE_QUARTER = QUARTER * 3
    MASK = FULL - 1
    EOF_SYMBOL = 256

    @staticmethod
    def build_frequencies(data: bytes) -> list[int]:
        """统计算术编码使用的符号频率。

        Args:
            data: 待编码的原始字节数据。

        Returns:
            长度为 257 的频率列表，前 256 项对应字节值，最后一项对应
            EOF 结束符号。
        """
        freq = [1] * 257
        for value in data:
            freq[value] += 1
        return freq

    @staticmethod
    def cumulative(freq: list[int]) -> list[int]:
        """根据频率表生成累积频率表。

        Args:
            freq: 符号频率列表。

        Returns:
            累积频率列表，长度比输入频率表多 1，用于区间划分。
        """
        total = 0
        cum = [0]
        for item in freq:
            total += item
            cum.append(total)
        return cum

    @classmethod
    def encode(cls, data: bytes) -> tuple[np.ndarray, dict]:
        """将字节数据编码为算术编码比特流。

        Args:
            data: 待编码的原始字节数据。

        Returns:
            一个二元组，包含 uint8 类型的编码比特流，以及解码所需的
            元数据字典。元数据中保存了符号频率表。
        """
        freq = cls.build_frequencies(data)
        cum = cls.cumulative(freq)
        total = cum[-1]
        low, high, pending = 0, cls.MASK, 0
        bits: list[int] = []

        def emit(bit: int) -> None:
            """输出一个确定比特并补齐延迟比特。

            Args:
                bit: 当前确定输出的比特值。
            """
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
        """根据算术编码比特流恢复原始字节数据。

        Args:
            bits: 算术编码得到的比特流。
            meta: 编码阶段保存的元数据，需包含 ``freq`` 频率表。

        Returns:
            解码恢复出的原始字节数据。
        """
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


def build_huffman_codes(data: bytes) -> dict[int, str]:
    """根据字节频率构建哈夫曼编码表。

    Args:
        data: 待编码的原始字节数据。

    Returns:
        字典形式的哈夫曼码表，键为字节值，值为对应的二进制码字字符串。
    """
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
        """递归遍历哈夫曼树并填充码表。

        Args:
            node: 当前遍历到的树节点。
            prefix: 从根节点到当前节点对应的码字前缀。
        """
        if isinstance(node, int):
            codes[node] = prefix or "0"
            return
        walk(node[0][2], prefix + "0")
        walk(node[1][2], prefix + "1")

    walk(heap[0][2], "")
    return codes


def build_shannon_fano_codes(data: bytes) -> dict[int, str]:
    """根据字节频率构建香农-范诺编码表。

    Args:
        data: 待编码的原始字节数据。

    Returns:
        字典形式的香农-范诺码表，键为字节值，值为对应的二进制码字字符串。
    """
    items = sorted(Counter(data).items(), key=lambda item: (-item[1], item[0]))
    codes = {symbol: "" for symbol, _ in items}

    def split(block: list[tuple[int, int]]) -> None:
        """递归划分符号集合并追加码字前缀。

        Args:
            block: 按频率排序后的符号与频率列表。
        """
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


def source_encode(data: bytes, method: str) -> tuple[np.ndarray, dict]:
    """对信源字节数据进行信源编码。

    Args:
        data: 来自信源预处理阶段的原始字节载荷。
        method: 信源编码方式。``"算术编码"`` 使用算术编码，
            ``"哈夫曼编码"`` 使用哈夫曼编码，其他取值使用香农-范诺编码。

    Returns:
        一个二元组，包含 uint8 类型的信源编码比特流，以及解码所需的
        元数据字典。
    """
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
    """将信源编码比特流解码回字节数据。

    Args:
        bits: 待解码的信源编码比特流，通常来自信道解码输出。
        meta: 信源编码阶段保存的元数据，用于恢复码表或频率表。
        method: 信源解码方式，需要与编码阶段使用的方式一致。

    Returns:
        解码恢复出的字节数据，后续会交给信源输出恢复模块展示。
    """
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
