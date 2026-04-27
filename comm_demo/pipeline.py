from __future__ import annotations

import io
import math
import time
import wave
import zlib
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
from PIL import Image

# 全局常量
DEFAULT_TEXT = "这是一个通信系统教学演示示例，可用于测试完整链路。"
SPS = 8             # 每符号采样点数
DEFAULT_ROLL_OFF = 0.35    # 根升余弦滤波器默认系数
DEFAULT_SPAN = 8           # 根升余弦滤波器默认跨度（多少个符号周期）
STAGE_NAMES = [
    "信源输入",      # 0: 读取文本/图像/音频数据
    "信源编码",      # 1: 数据压缩（哈夫曼/算术/香农-范诺）
    "信道编码",      # 2: 差错控制编码（CRC/汉明/卷积）
    "成型调制",      # 3: 数字调制+脉冲成型（MASK/MPSK/MQAM）
    "信道传输",      # 4: 通过噪声信道（AWGN/瑞利/莱斯衰落）
    "匹配滤波与判决", # 5: 接收端滤波、采样、判决
    "信道解码",      # 6: 译码纠错
    "信源解码",      # 7: 解压缩恢复原始数据
]

"""
信源数据容器：存储原始输入数据及其元信息

属性:
    kind: 数据类型 ("文本"/"图像"/"语音")
    raw_bytes: 原始字节数据（统一编码后的二进制）
    preview_text: 用于UI显示的预览文本
    text_value: 文本模式下的字符串内容
    image_array: 图像模式下的像素数组（灰度图）
    audio_samples: 语音模式下的采样数组（float32）
    sample_rate: 语音采样率（Hz）
    audio_wav_bytes: 语音的WAV格式字节（用于播放）
"""
@dataclass
class SourceArtifact:
    kind: str
    raw_bytes: bytes
    preview_text: str
    text_value: str = ""
    image_array: np.ndarray | None = None
    audio_samples: np.ndarray | None = None
    sample_rate: int = 0
    audio_wav_bytes: bytes | None = None


"""
    仿真配置参数：用户设置的所有仿真参数

    属性:
        kind: 输入数据类型
        text: 文本输入内容
        path: 文件路径（图像/语音模式）
        source_method: 信源编码方法（"无"/"哈夫曼编码"/"算术编码"/"香农-范诺编码"）
        channel_method: 信道编码方法（"无"/"CRC"/"汉明码"/"卷积码"）
        modulation: 调制方式（"MASK"/"MPSK"/"MQAM"）
        order: 调制阶数（如4表示4PSK、16表示16QAM）
        channel_name: 信道模型（"AWGN"/"瑞利衰落"/"莱斯衰落"）
        snr_db: 信噪比（dB）
        k_factor: 莱斯衰落K因子（直射径与散射径功率比）
        roll_off: 滚降系数
"""
@dataclass
class SimulationConfig:
    kind: str
    text: str
    path: str
    source_method: str
    channel_method: str
    modulation: str
    order: int
    channel_name: str
    snr_db: float
    k_factor: float
    roll_off: float = DEFAULT_ROLL_OFF
    gray_ok: bool = False

"""
    完整仿真结果：包含链路各阶段的信号和数据
    存储了从信源输入到最终恢复的完整数据流，用于分析和可视化展示
"""
@dataclass
class SimulationResult:
    config: SimulationConfig   # 仿真配置
    source: SourceArtifact     # 原始信源数据
    source_bits: np.ndarray    # 原始比特流
    source_coded_bits: np.ndarray  # 信源编码后的比特流
    channel_coded_bits: np.ndarray  # 信道编码后的比特流
    tx_symbols: np.ndarray          # 调制后的复数信号--星座点
    tx_signal: np.ndarray           # 上采样＋脉冲成型后的发送信号
    pulse: np.ndarray               # 根升余弦脉冲成型滤波器系数
    rx_signal: np.ndarray           # 经过信道（噪声＋衰落）后的接收信号
    matched_signal: np.ndarray      # 匹配滤波后的信号
    sampled_symbols: np.ndarray     # 采样后的符号（用于星座点图显示）
    detected_symbols: np.ndarray    # 判决后的符号（星座点）
    rx_channel_bits: np.ndarray     # 解调后的比特流
    decoded_source_bits: np.ndarray  # 信道解码后的比特流
    restored_bytes: bytes            # 信源解码恢复的字节数据
    restored_text: str               # 恢复的文本信息
    restored_image: np.ndarray | None  # 恢复的图像数组
    restored_audio_samples: np.ndarray | None   # 恢复的语言采样
    restored_audio_rate: int           # 恢复的语言采样率
    restored_audio_wav_bytes: bytes | None   # 恢复的WAV格式语言
    ber: float                           # 误比特率
    history: list[str] = field(default_factory=list)
    crc_ok: bool | None = None          # 是否使用CRC
    gray_ok:bool | None = None          # 是否使用格雷编码
    stage_times: dict[str, float] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """返回仿真摘要文本（用于UI显示）"""
        return self.summary_text()

    @property
    def src_bits(self) -> np.ndarray:
        """返回原始比特流（兼容旧接口）"""
        return self.source_bits

    @property
    def preview(self) -> str:
        """返回恢复内容的预览文本"""
        return self.restored_text

    def summary_text(self) -> str:
        """
        生成格式化的仿真结果摘要
        包含：输入类型、编码方式、调制参数、信道条件、性能指标等
        """
        lines = [
            f"输入类型: {self.config.kind}",
            f"信源编码: {self.config.source_method}",
            f"信道编码: {self.config.channel_method}",
            f"调制方式: {self.config.modulation}-{self.config.order}",
            f"信道模型: {self.config.channel_name}",
            f"SNR: {self.config.snr_db:.1f} dB",
            f"滚降系数: {self.config.roll_off:.2f}",
            f"原始比特数: {len(self.source_bits)}",
            f"信源编码后比特数: {len(self.source_coded_bits)}",
            f"信道编码后比特数: {len(self.channel_coded_bits)}",
            f"BER: {self.ber:.6f}",
        ]
        if self.crc_ok is not None:
            lines.append(f"CRC校验: {'通过' if self.crc_ok else '失败'}")
        return "\n".join(lines)

    """
        返回各阶段的比特流视图（用于比特流可视化）
        展示数据在链路中如何被处理和变换
    """
    def bit_views(self) -> dict[str, np.ndarray]:
        return {
            "原始比特流": self.source_bits,
            "信源编码后": self.source_coded_bits,
            "信道编码后": self.channel_coded_bits,
            "判决后比特流": self.rx_channel_bits,
            "信道解码后": self.decoded_source_bits,
        }

    """
        返回各阶段的信号视图（用于波形/频谱可视化）
        展示信号在时域的波形变化
    """
    def signal_views(self) -> dict[str, np.ndarray]:
        return {
            "调制后信号": self.tx_signal,
            "信道后信号": self.rx_signal,
            "匹配滤波后信号": self.matched_signal,
        }

"""
    分步仿真会话：支持逐步执行和状态保存
    与SimulationResult不同，Session用于交互式分步演示，
    每一步执行一个通信阶段，可以暂停查看中间结果
"""
@dataclass
class SimulationSession:
    config: SimulationConfig
    stage_index: int = 0  # 当前执行到的阶段索引
    history: list[str] = field(default_factory=list)  # 执行历史

    # 各阶段的中间结果（逐步填充）
    source: SourceArtifact | None = None
    source_bits: np.ndarray | None = None
    source_coded_bits: np.ndarray | None = None
    source_meta: dict | None = None
    channel_coded_bits: np.ndarray | None = None
    channel_meta: dict | None = None
    tx_symbols: np.ndarray | None = None
    tx_signal: np.ndarray | None = None
    pulse: np.ndarray | None = None
    rx_signal: np.ndarray | None = None
    fading_symbols: np.ndarray | None = None
    matched_signal: np.ndarray | None = None
    sampled_symbols: np.ndarray | None = None
    detected_symbols: np.ndarray | None = None
    rx_channel_bits: np.ndarray | None = None
    decoded_source_bits: np.ndarray | None = None
    restored_bytes: bytes | None = None
    stage_times: dict[str, float] = field(default_factory=dict)

    def is_finished(self) -> bool:
        """检查是否已完成所有阶段"""
        return self.stage_index >= len(STAGE_NAMES)

    def next_stage(self) -> str:
        """返回下一个阶段的名称（或'已完成'）"""
        return "已完成" if self.is_finished() else STAGE_NAMES[self.stage_index]


    """
        执行下一个仿真阶段
        根据stage_index执行对应的处理函数，并更新状态
        返回执行结果的描述字符串
    """
    def step(self) -> str:
        stage_name = self.next_stage()
        tick = time.perf_counter()
        if self.stage_index == 0:
            self.source = prepare_source(self.config.kind, self.config.text, self.config.path)
            self.source_bits = bytes_to_bits(self.source.raw_bytes)
            message = f"完成信源输入：{self.config.kind}，原始比特数 {len(self.source_bits)}"
        elif self.stage_index == 1:
            self.source_coded_bits, self.source_meta = source_encode(self.source.raw_bytes, self.config.source_method)
            message = f"完成信源编码：{self.config.source_method}，输出比特数 {len(self.source_coded_bits)}"
        elif self.stage_index == 2:
            self.channel_coded_bits, self.channel_meta = channel_encode(self.source_coded_bits, self.config.channel_method)
            message = f"完成信道编码：{self.config.channel_method}，输出比特数 {len(self.channel_coded_bits)}"
        elif self.stage_index == 3:
            self.tx_signal, self.tx_symbols, self.pulse = modulate(
                self.channel_coded_bits, self.config.modulation, self.config.order, self.config.roll_off, self.config.gray_ok
            )
            message = f"完成成型调制：{self.config.modulation}-{self.config.order}"
        elif self.stage_index == 4:
            self.rx_signal, self.fading_symbols = apply_channel(
                self.tx_signal,
                self.tx_symbols,
                self.config.channel_name,
                self.config.snr_db,
                self.config.k_factor,
            )
            message = f"完成信道传输：{self.config.channel_name}"
        elif self.stage_index == 5:
            self.matched_signal, self.sampled_symbols, self.detected_symbols, self.rx_channel_bits = demodulate(
                self.rx_signal,
                self.pulse,
                self.fading_symbols,
                self.config.modulation,
                self.config.order,
                len(self.channel_coded_bits),
                self.config.gray_ok,
            )
            message = f"完成匹配滤波与判决：恢复比特数 {len(self.rx_channel_bits)}"
        elif self.stage_index == 6:
            self.decoded_source_bits, crc_ok = channel_decode(
                self.rx_channel_bits, self.channel_meta, self.config.channel_method
            )
            if crc_ok is not None:
                self.channel_meta["crc_ok"] = crc_ok
            message = f"完成信道解码：输出比特数 {len(self.decoded_source_bits)}"
        elif self.stage_index == 7:
            self.restored_bytes = source_decode(self.decoded_source_bits, self.source_meta, self.config.source_method)
            message = "完成信源解码与数据恢复"
        else:
            return "已完成"
        elapsed = time.perf_counter() - tick
        self.stage_times[stage_name] = elapsed
        print(f"[pipeline] {stage_name}: {elapsed * 1000:.2f} ms")
        self.stage_index += 1
        self.history.append(f"{self.stage_index}. {message}")
        return self.history[-1]

    def run_all(self) -> SimulationResult:
        while not self.is_finished():
            self.step()
        return self.build_result()

    def build_result(self) -> SimulationResult:
        restored_text, restored_image, restored_audio_samples, restored_audio_rate, restored_audio_wav = restore_output(
            self.source.kind, self.restored_bytes or b""
        )
        compare_len = min(len(self.source_coded_bits), len(self.decoded_source_bits))
        ber = float(np.mean(self.source_coded_bits[:compare_len] != self.decoded_source_bits[:compare_len])) if compare_len else 0.0
        return SimulationResult(
            config=self.config,
            source=self.source,
            source_bits=self.source_bits,
            source_coded_bits=self.source_coded_bits,
            channel_coded_bits=self.channel_coded_bits,
            tx_symbols=self.tx_symbols,
            tx_signal=self.tx_signal,
            pulse=self.pulse,
            rx_signal=self.rx_signal,
            matched_signal=self.matched_signal,
            sampled_symbols=self.sampled_symbols,
            detected_symbols=self.detected_symbols,
            rx_channel_bits=self.rx_channel_bits,
            decoded_source_bits=self.decoded_source_bits,
            restored_bytes=self.restored_bytes or b"",
            restored_text=restored_text,
            restored_image=restored_image,
            restored_audio_samples=restored_audio_samples,
            restored_audio_rate=restored_audio_rate,
            restored_audio_wav_bytes=restored_audio_wav,
            ber=ber,
            history=self.history.copy(),
            crc_ok=self.channel_meta.get("crc_ok") if self.channel_meta else None,
            stage_times=self.stage_times.copy(),
        )


"""
    算术编码器/解码器实现类。

    使用 32 位整数定点运算模拟区间划分，避免了浮点数精度问题。
    包含特殊的 EOF（文件结束）符号处理，支持自适应流式解码。
"""
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
    

"""
   将 PIL 图像对象打包为带尺寸头部的二进制数据流，并返回灰度像素矩阵。
   该函数首先将输入图像转换为灰度图，然后将图像尺寸信息（宽、高）编码为
   8字节的头部数据，最后拼接像素数据生成可用于传输的字节流。

   Args:
       image (Image.Image): 输入的 PIL 图像对象（可以是彩色或灰度）。

   Returns:
       tuple[bytes, np.ndarray]: 包含两个元素的元组：
           - bytes: 打包后的二进制数据流（前8字节为宽高信息，后续为像素数据）。
           - np.ndarray: 转换后的灰度像素矩阵，形状为 (高, 宽)，数据类型为 uint8。
"""
def pack_image_data(image: Image.Image) -> tuple[bytes, np.ndarray]:
    gray = image.convert("L")    # 将图片转换为灰度图
    pixels = np.array(gray, dtype=np.uint8)   # 将灰度图像转换成二维矩阵
    """生成图片大小的8字节的头部信息"""
    header = gray.size[0].to_bytes(4, "big") + gray.size[1].to_bytes(4, "big")
    return header + pixels.tobytes(), pixels


def unpack_image_data(data: bytes) -> np.ndarray:
    if len(data) < 8:
        raise ValueError("图像数据长度不足。")
    width = int.from_bytes(data[:4], "big")
    height = int.from_bytes(data[4:8], "big")
    pixels = np.frombuffer(data[8:], dtype=np.uint8)
    if len(pixels) != width * height:
        raise ValueError("图像数据损坏。")
    return pixels.reshape(height, width).copy()


def pack_audio_data(samples: np.ndarray, sample_rate: int) -> bytes:
    mono = samples.astype(np.int16)
    quantized = np.clip(np.round(mono / 256.0), -128, 127).astype(np.int8)
    header = sample_rate.to_bytes(4, "big") + len(quantized).to_bytes(4, "big")
    return header + quantized.tobytes()


def unpack_audio_data(data: bytes) -> tuple[int, np.ndarray]:
    if len(data) < 8:
        raise ValueError("语音数据长度不足。")
    sample_rate = int.from_bytes(data[:4], "big")
    count = int.from_bytes(data[4:8], "big")
    payload = data[8:]
    quantized = np.frombuffer(payload[:count], dtype=np.int8).copy()
    if len(quantized) != count:
        raise ValueError("语音数据损坏。")
    samples = (quantized.astype(np.int16) * 256).astype(np.int16)
    return sample_rate, samples


def pcm_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


# ==================== 信源输入处理 ====================
def prepare_source(kind: str, text: str, path: str) -> SourceArtifact:
    """
    准备信源数据：根据类型（文本/图像/语音）读取并打包数据
    统一转换为字节流，便于后续数字处理
    """
    if kind == "文本":
        content = (text.strip() or DEFAULT_TEXT).replace("\r\n", "\n").replace("\r", "\n")
        return SourceArtifact(kind=kind, raw_bytes=content.encode("utf-8"), preview_text=content, text_value=content)
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError("请先选择有效文件。")
    if kind == "图像":
        image = Image.open(file_path)
        raw_bytes, pixels = pack_image_data(image) # raw_bytes = 图片二进制流 pixels = 图片像素矩阵
        return SourceArtifact(
            kind=kind,
            raw_bytes=raw_bytes,
            preview_text=f"{file_path.name}\n尺寸: {pixels.shape[1]} x {pixels.shape[0]}",
            image_array=pixels,
        )
    with wave.open(str(file_path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError("当前仅支持 16-bit PCM WAV 音频。")
        frames = wav_file.readframes(wav_file.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
        if wav_file.getnchannels() > 1:
            samples = samples.reshape(-1, wav_file.getnchannels())[:, 0]
        payload = pack_audio_data(samples, wav_file.getframerate())
        return SourceArtifact(
            kind=kind,
            raw_bytes=payload,
            preview_text=f"{file_path.name}\n采样率: {wav_file.getframerate()} Hz\n时长: {len(samples) / wav_file.getframerate():.2f} s",
            audio_samples=samples.astype(np.float32),
            sample_rate=wav_file.getframerate(),
            audio_wav_bytes=pcm_to_wav_bytes(samples, wav_file.getframerate()),
        )


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

"""
    对比特流进行 (7, 4) 汉明编码。

    编码原理：将每 4 位数据位编码为 7 位码字（新增 3 位校验位）。
    这种编码能够检测并纠正单个比特的错误。

    编码结构（输出顺序）：[P1, P2, D1, P4, D2, D3, D4]
    - D1-D4: 原始数据位。
    - P1, P2, P4: 计算得出的校验位。

    Args:
        bits (np.ndarray): 原始比特流数组。

    Returns:
        np.ndarray: 编码后的比特流数组。长度变为原来的 7/4 倍（向上取整）。
"""
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
    # Deterministic pseudo-random binary sequence for data whitening.
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


def apply_channel(
    tx_signal: np.ndarray,
    tx_symbols: np.ndarray,
    channel_name: str,
    snr_db: float,
    k_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    fading = np.ones(len(tx_symbols), dtype=np.complex64)
    if channel_name == "瑞利衰落":
        fading = ((rng.normal(size=len(tx_symbols)) + 1j * rng.normal(size=len(tx_symbols))) / np.sqrt(2)).astype(np.complex64)
    elif channel_name == "莱斯衰落":
        los = math.sqrt(k_factor / (k_factor + 1))
        scatter = math.sqrt(1 / (k_factor + 1))
        fading = (
            los + scatter * (rng.normal(size=len(tx_symbols)) + 1j * rng.normal(size=len(tx_symbols))) / np.sqrt(2)
        ).astype(np.complex64)
    fading_samples = np.repeat(fading, SPS)
    if len(fading_samples) < len(tx_signal):
        fading_samples = np.r_[fading_samples, np.repeat(fading[-1], len(tx_signal) - len(fading_samples))]
    rx = tx_signal * fading_samples[: len(tx_signal)]
    power = np.mean(np.abs(rx) ** 2, dtype=np.float64) + 1e-12
    noise_power = power * SPS / (10 ** (snr_db / 10))
    noise = rng.normal(scale=np.sqrt(noise_power / 2), size=len(rx)) + 1j * rng.normal(
        scale=np.sqrt(noise_power / 2), size=len(rx)
    )
    return (rx + noise).astype(np.complex64, copy=False), fading.astype(np.complex64, copy=False)

    """
    判决器：在星座图中寻找与均衡信号欧几里得距离最近的星座点。
    
    采用分块处理机制，防止在长序列仿真时因距离矩阵过大导致内存溢出。
    
    Args:
        equalized: 均衡后的复数采样点序列 
        points: 标准星座图上的复数坐标集合
        
    Returns:
        与输入序列等长的整数索引数组，代表每个采样点被判为哪个星座点
    """
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
    equalized = (sampled / np.where(np.abs(fading) < 1e-8, 1.0 + 0j, fading)).astype(np.complex64)
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
    return matched, sampled, points[detected_indices], bits


def restore_output(kind: str, restored_bytes: bytes) -> tuple[str, np.ndarray | None, np.ndarray | None, int, bytes | None]:
    if kind == "文本":
        return restored_bytes.decode("utf-8", errors="replace"), None, None, 0, None
    if kind == "图像":
        try:
            image = unpack_image_data(restored_bytes)
            return f"图像恢复成功\n尺寸: {image.shape[1]} x {image.shape[0]}", image, None, 0, None
        except Exception:
            return "图像恢复失败，可能由信道误码导致。", None, None, 0, None
    try:
        sample_rate, samples = unpack_audio_data(restored_bytes)
        wav_bytes = pcm_to_wav_bytes(samples, sample_rate)
        return f"语音恢复完成\n采样率: {sample_rate} Hz\n采样点数: {len(samples)}", None, samples.astype(np.float32), sample_rate, wav_bytes
    except Exception:
        return "语音恢复失败，输出数据为空或已损坏。", None, None, 0, None


def create_session(
    kind: str,
    text: str,
    path: str,
    source_method: str,
    channel_method: str,
    modulation: str,
    order: int,
    channel_name: str,
    snr_db: float,
    k_factor: float,
    roll_off: float = DEFAULT_ROLL_OFF,
    gray_ok: bool = False,
) -> SimulationSession:
    return SimulationSession(
        SimulationConfig(
            kind, text, path, source_method, channel_method, modulation, order, channel_name, snr_db, k_factor, roll_off, gray_ok
        )
    )


def run_pipeline(
    kind: str,
    text: str,
    path: str,
    source_method: str,
    channel_method: str,
    modulation: str,
    order: int,
    channel_name: str,
    snr_db: float,
    k_factor: float,
    roll_off: float = DEFAULT_ROLL_OFF,
    gray_ok: bool = False,
) -> SimulationResult:
    return create_session(
        kind, text, path, source_method, channel_method, modulation, order, channel_name, snr_db, k_factor, roll_off, gray_ok
    ).run_all()


def simulate_raw_modem(
    bits: np.ndarray,
    modulation: str,
    order: int,
    channel_name: str,
    snr_db: float,
    k_factor: float = 3.0,
    roll_off: float = DEFAULT_ROLL_OFF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tx_signal, tx_symbols, pulse = modulate(bits.astype(np.uint8), modulation, order, roll_off)
    rx_signal, fading_symbols = apply_channel(tx_signal, tx_symbols, channel_name, snr_db, k_factor)
    matched_signal, sampled_symbols, _, detected_bits = demodulate(
        rx_signal, pulse, fading_symbols, modulation, order, len(bits)
    )
    return detected_bits, sampled_symbols, matched_signal
