from __future__ import annotations # 兼容Python不同版本的类型提示

import io # 用于处理内存中的二进制数据
import math # 用于数学计算，如计算音频帧数
import wave # 用于处理WAV音频文件
import zlib # 用于压缩和解压缩数据
from bisect import bisect_right # 用于在有序列表中查找插入位置
from collections import Counter # 用于计数对象的出现次数
from dataclasses import dataclass, field # 用于定义数据类
from heapq import heappush, heappop # 用于实现优先队列
from pathlib import Path # 用于处理文件路径

import numpy as np # 用于数值计算和数组操作
from PIL import Image # 用于图像处理

# 全局常量
DEFAULT_TEXT = "这是一个通信系统教学演示示例，可用于测试完整链路。"
SPS = 8    # 每符号采样点数
DEFAULT_ROLL_OFF = 0.35 # 滚降系数
DEFAULT_SPAN = 8     # 滤波器跨度（符号数） 时域无限长，需要截断
STAGE_NAME = [
    "信源输入",
    "信源编码",
    "信道编码",
    "成型调制",
    "信道传输",
    "匹配滤波与判决",
    "信道解码",
    "信源解码",
]

"""
定义输入数据的数据类，包含原始字节数据、预览文本、文本内容、图像数组、音频采样值等字段。
"""
@dataclass
class SourceArtifact:
    kind: str # 输入数据类型，如"text"、"image"或"audio"
    raw_bytes: bytes # 原始字节数据
    preview_text: str = "" # 用于UI显示的文本
    text_vlaue: str = "" # 文本模式下的字符串内容
    image_arraay:np.ndarray | None = None # 图像模式下的NumPy数组
    audio_samples: np.ndarray | None = None # 音频模式下的采样值数组
    sample_rate: int = 0 # 音频采样率
    audio_wave_bytes: bytes | None = None # 音频模式下的WAV格式字节数据

"""
定义仿真配置的数据类，包含通信系统仿真所需的各种参数。
"""
@dataclass
class SimulationConfig:
    kind: str # 输入数据类型
    text: str # 输入文本
    path: str # 输入文件路径
    source_method: str # 信源编码方法
    channel_method: str # 信道编码方法
    modulation: str # 调制方式
    order: int # 调制阶数
    channel_name: str # 信道模型
    snr_db: float # 信噪比（dB）
    k_factor: float # 莱斯衰落的K因子(直射路径功率与散射路径功率之比)
    roll_off: float = DEFAULT_ROLL_OFF # 滚降系数

"""
定义仿真结果的数据类，包含仿真过程中产生的各种数据和结果，如信源编码后的比特流、调制后的信号、接收信号、解调后的比特流、恢复的文本内容等。
"""
@dataclass
class SimulationResult:
    config: SimulationConfig # 仿真配置
    source: SourceArtifact # 输入数据的处理结果
    source_bits: np.ndarray # 原始数据比特流
    source_coded_bits: np.ndarray # 信源编码后的比特流
    channel_coded_bits: np.ndarray # 信道编码后的比特流
    tx_symbols: np.ndarray # 调制后的复数信号--星座点
    tx_signal: np.ndarray # 成型滤波后的时域信号
    pulse: np.ndarray # 根升余弦脉冲成型滤波器系数
    rx_signal: np.ndarray # 经过信道(噪声＋衰落)后的接收信号
    matched_signal: np.ndarray # 匹配滤波后的信号
    sampled_symbols: np.ndarray # 采样后的符号
    detected_symbols: np.ndarray # 判决后的符号
    rx_channel_bits: np.ndarray # 解调后的比特流
    decoded_source_bits: np.ndarray # 信道解码后的比特流
    restored_bytes: bytes # 信源解码恢复的字节数据
    restored_text: str # 信源解码恢复的文本内容
    restored_image: np.ndarray | None # 信源解码恢复的图像数组
    restored_audio_samples: np.ndarray | None # 信源解码恢复的音频采样值
    restored_audio_rate: int # 信源解码恢复的音频采样率
    restored_audio_wav_bytes: bytes | None # 信源解码恢复的音频WAV字节数据
    ber: float # 误比特率
    history: list[str] = field(default_factory=list) # 仿真过程中的历史记录，用于UI显示
    crc_ok: bool | None = None # CRC校验结果，True表示校验通过，False表示校验失败，None表示未进行CRC校验

    """返回仿真结果的摘要文本(用于UI显示)"""
    @property
    def summary(self) -> str:
        return self.summary_text()
    
    """返回原始比特流"""
    @property
    def src_bits(self) -> np:
        return self.source_bits
    
    """返回解码后的文本内容"""
    @property
    def preview(self) -> str:
        return self.restored_text
    
    """生成仿真结果的摘要文本,包含输入数据类型、信道模型、调制方式、SNR、误比特率等信息"""
    def summary_text(self) -> str:
        lines = [
            f"输入类型: {self.config.kind}",
            f"信源编码: {self.config.source_method}",
            f"信道编码: {self.config.channel_method}",
            f"调制方式: {self.config.modulation}-{self.config.order}",
            f"信道模型: {self.config.channel_name}",
            f"SNR: {self.config.snr_db:.1f} dB",
            f"滚降系数: {self.config.roll_off:.2f}",
            f"原始比特数：{len(self.source_bits)}",
            f"信源编码后比特数：{len(self.source_coded_bits)}",
            f"信道编码后比特数：{len(self.channel_coded_bits)}",
            f"BER: {self.ber:.6f}"
        ]
        if self.crc_ok is not None:
            lines.append(f"CRC校验: {'通过' if self.crc_ok else '失败'}")
        return "\n".join(lines)
    
    """返回不同阶段的比特流视图"""
    def bit_views(self) -> dict[str, np.ndarray]:
        return{
            "原始比特流": self.source_bits,
            "信源编码后比特流": self.source_coded_bits,
            "信道编码后比特流": self.channel_coded_bits,
            "判决后的比特流": self.rx_channel_bits,
            "信道解码后的比特流": self.decoded_source_bits,
        }

    """返回不同阶段的信号视图"""
    def signal_views(self) -> dict[str, np.ndarray]:
        return {
            "调制后的信号": self.tx_signal,
            "信道后信号": self.rx_signal,
            "匹配滤波后信号": self.matched_signal,
        }
    

@dataclass
class SimulationSession:
    config: SimulationConfig # 仿真配置
    stage_index: int = 0 # 当前阶段索引
    history: list[str] = field(default_factory=list) # 仿真过程中的历史记录

    source: SourceArtifact | None = None # 输入数据的处理结果
    source_bits: np.ndarray | None = None # 原始数据比特流
    source_coded_bits: np.ndarray | None = None # 信源编码后的比特流
    source_meta: dict | None = None # 信源编码的元信息，如编码参数等
    channel_coded_bits: np.ndarray | None = None # 信道编码后的比特流
    channel_meta: dict | None = None # 信道编码的元信息，如编码参数等
    tx_symbols: np.ndarray | None = None # 调制后的复数信号--星座点
    tx_signal: np.ndarray | None = None # 成型滤波后的时域信号
    pulse: np.ndarray | None = None # 根升余弦脉冲成型滤波器系数
    rx_signal: np.ndarray | None = None # 经过信道(噪声＋衰落)后的接收信号
    fading_symmbols: np.ndarray | None = None # 衰落系数
    matched_signal: np.ndarray | None = None # 匹配滤波后的信号
    sampled_symbols: np.ndarray | None = None # 采样后的符号
    detected_symbols: np.ndarray | None = None # 判决后的符号
    rx_channel_bits: np.ndarray | None = None # 解调后的比特流
    decoded_source_bits: np.ndarray | None = None # 信道解码后的比特流
    restored_bytes: bytes | None = None # 信源解码恢复的字节数据

    """检查是否已完成所有阶段"""
    def is_finished(self) -> bool:
        return self.stage_index >= len(STAGE_NAME)
    
    """返回下一个阶段的名称，如果已完成则返回'已完成'"""
    def next_stage(self) -> str:
        return "已完成" if self.is_finished() else STAGE_NAME[self.stage_index]
    
    def step(self) -> str:
        if self.stage_index == 0:
            self.source = prepare_source(self.config.kind, self.config.text, self.config.path)
            self.source_bits = bytes_to_bits(self.source.raw_bytes)
            message = f"完成信源输入，{self.config.kind}, 原始比特数: {len(self.source_bits)}"
        elif self.stage_index == 1:
            self.source_coded_bits, self.source_meta = source_encode(self.source.raw_bytes, self.config.source_method)
            message = f"完成信源编码，编码方法: {self.config.source_method}, 编码后比特数: {len(self.source_coded_bits)}"
        elif self.stage_index == 2:
            self.channel_coded_bits, self.channel_meta = channel_encode(self.source_coded_bits, self.config.channel_method)
            message = f"完成信道编码，编码方法: {self.config.channel_method}, 编码后比特数: {len(self.channel_coded_bits)}"
        elif self.stage_index == 3:
            self.tx_singal, self.tx_symbols, self.pulse = modulate(
                self.channel_coded_bits, self.config.modulation, self.config.order, self.config.roll_off
            )
            message = f"完成成型调制：{self.config.modulation}-{self.config.order}"
        elif self.stage_index == 4:
            self.rx_signal, self.fading_symbols = apply_channel(
                self.tx_singal,
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
        self.stage_index += 1
        self.history.append(f"{self.stage_index}. {message}")
        return self.history[-1] # 返回当前阶段的完成信息
    
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
        )
            

"""根据输入的类型、文本内容和文件路径准备信源数据，返回一个SourceArtifact对象。"""
def prepare_source(kind: str, text: str, path: str) -> SourceArtifact:
    if kind == "文本":
        content = (text.strip() or DEFAULT_TEXT).replace("\r\n", "\n").replace("\r", "\n")
        return SourceArtifact(
            kind=kind,
            raw_bytes=content.encode("utf-8"),
            preview_text=content,
            text_vlaue=content,
        )
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError("请先选择有效文件。")
    if kind == "图像":
        image = Image.open(file_path)
        raw_bytes, pixels = pack_image_data(image) # # raw_bytes = 图片二进制流 pixels = 图片像素矩阵
        return SourceArtifact(
            kind=kind,
            raw_bytes=raw_bytes,
            preview_text=f"{file_path.name}\n尺寸:({pixels.shape[1]}x{pixels.shape[0]})",
            image_arraay=pixels,
        )
    with wave.open(str(file_path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError("当前仅支持16-bit PCM格式的WAV文件。")
        frames = wav_file.readframes(wav_file.getnframes()) # 读取音频帧数据
        samples = np.frombuffer(frames, dtype=np.int16) # 将每两个字节转为16bit整数，形成一维的采样值数组
        if wav_file.getnchannels() > 1:
            samples = samples.reshape(-1, wav_file.getnchannels())[:, 0] # 只使用第一声道
        payload = pack_audio_data(samples, wav_file.getframerate()) # 将采样值和采样率打包成字节流
        return SourceArtifact(
            kind=kind,
            raw_bytes=payload,
            preview_text=f"{file_path.name}\n采样率: {wav_file.getframerate()} Hz\n时长: {len(samples) / wav_file.getframerate():.2f} 秒",
            audio_samples=samples.astype(np.float32),
            sample_rate=wav_file.getframerate(),
            audio_wave_bytes=pcm_to_wav_bytes(samples, wav_file.getframerate()),
        )



"""将图像数据打包成字节流，前8字节存储图像宽高，后续字节存储像素数据"""
def pack_image_data(image: Image.Image) -> tuple[bytes, np.ndarray]:
    gray = image.convert("L") # 转为灰度图
    pixels = np.array(gray, dtype=np.uint8) # 转为NumPy数组
    deader = gray.size[0].to_bytes(4, "big") + gray.size[1].to_bytes(4, "big") # 前8字节存储图像宽高
    return deader + pixels.tobytes(), pixels

"""将字节数据转换回图像数据，输入是包含图像宽高和像素数据的字节流，返回一个NumPy数组表示的图像像素矩阵"""
def unpack_image_data(data: bytes) -> np.ndarray:
    if len(data) < 8:
        raise ValueError("图像数据长度不足。")
    width = int.from_bytes(data[:4], "big")
    height = int.from_bytes(data[4:8], "big")
    pixels = np.frombuffer(data[8:], dtype=np.uint8)
    if len(pixels) != width * height:
        raise ValueError("图像数据损坏。")
    return pixels.reshape(height, width).copy()

"""将音频数据打包成字节流，前8字节存储采样率和采样值数量，后续字节存储量化后的采样值"""
def pack_audio_data(samples: np.ndarray, sample_rate: int) -> bytes:
    mono = samples.astype(np.int16) # 确保是16-bit整数
    quantized = np.clip(np.round(mono / 256.0), -128, 127).astype(np.int8) # 量化到8-bit范围
    header = sample_rate.to_bytes(4, "big") + len(quantized).to_bytes(4, "big") # 前8字节存储采样率和采样值数量
    return header + quantized.tobytes()

"""将字节数据转换回音频数据，输入是包含采样率、采样值数量和量化采样值的字节流，返回采样率和一个NumPy数组表示的音频采样值"""
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

"""将16-bit PCM格式的采样值转换为WAV格式的字节流，包含WAV文件头和音频数据"""
def pcm_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO() # 创建一个内存中的字节流对象
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1) # 单声道
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(sample_rate) # 采样率
        wav_file.writeframes(samples.astype(np.int16).tobytes()) # 写入音频数据
    return buffer.getvalue()

"""将字节数据转换为比特流，返回一个NumPy数组，每个元素是0或1"""
def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)) if data else np.zeros(0, dtype=np.uint8)

"""将比特流转换回字节数据，输入是一个NumPy数组，每个元素是0或1，返回对应的字节数据"""
def bits_to_bytes(bits: np.ndarray) -> bytes:
    if len(bits) == 0:
        return b""
    pad = (-len(bits)) % 8
    if pad:
        bits = np.r_[bits, np.zeros(pad, dtype=np.uint8)]
    return np.packbits(bits.astype(np.uint8)).tobytes()

"""将整数列表转换为比特流，输入是一个整数列表和每个整数的位宽，返回一个NumPy数组，每个元素是0或1"""
def ints_to_bits(values: list[int], width: int) -> np.ndarray:
    out: list[int] = []
    for value in values:
        out.extend([(value >> shift) & 1 for shift in range(width - 1, -1, -1)])
    return np.array(out, dtype=np.uint8)

"""将比特流转换回整数列表，输入是一个NumPy数组，每个元素是0或1，以及每个整数的位宽，返回对应的整数列表"""
def bits_to_ints(bits: np.ndarray, width: int) -> list[int]:
    if len(bits) % width:
        bits = np.r_[bits, np.zeros((-len(bits)) % width, dtype=np.uint8)]
    values = []
    for index in range(0, len(bits), width):
        value = 0
        for bit in bits[index : index + width]:
            value = (value << 1) | int(bit)
        values.append(value)
    return values

"""实现算术编码和解码的类，包含编码和解码方法，以及构建频率表和累积频率的方法。"""
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

"""实现信源编码，支持算术编码、霍夫曼编码和香农-范诺编码三种方法，输入是原始字节数据和编码方法名称，输出是编码后的比特流和编码的元信息（如编码表等）。"""
def source_encode(data: bytes, method: str) -> tuple[np.ndarray, dict]:
    if method == "算术编码":
        bits, meta = ArithmeticCoder.encode(data)
        meta["method"] = method  # 记录信源编码方法
        return bits, meta
    codes = build_huffman_codes(data) if method == "霍夫曼编码" else build_shannon_fano_codes(data)
    bits = np.array([int(bit) for byte in data for bit in codes[byte]], dtype=np.uint8) # 将每个字节转换为对应的二进制字符串编码，并展开成比特流
    return bits, {"method": method, "codes": codes, "length": len(data)} # 返回编码后的比特流和编码表等元信息

"""构建霍夫曼编码表，输入是字节数据，输出是一个字典，键是符号（0-255），值是对应的二进制字符串编码。"""
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

"""构建香农-范诺编码表，输入是字节数据，输出是一个字典，键是符号（0-255），值是对应的二进制字符串编码。"""
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

def channel_encode(bits: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    if method == "CRC":
        crc = zlib.crc32(bits_to_bytes(bits)) & 0xFF # 计算CRC-32校验值，并取最低8位作为CRC码
        return np.r_[bits, ints_to_bits([crc], 8)], {"length": len(bits)} # 将CRC码转换为比特流并附加到原始比特流后面，返回编码后的比特流和元信息（如原始比特长度等）
    if method == "汉明码":
        return hamming74_encode(bits), {"length": len(bits)} # 使用(7,4)汉明码进行编码，返回编码后的比特流和元信息（如原始比特长度等）
    return convolutional_encode(bits), {"length": len(bits)} # 使用卷积码进行编码，返回编码后的比特流和元信息（如原始比特长度等）

def channel_decode(bits: np.ndarray, meta: dict, method: str) -> tuple[np.ndarray, bool | None]:
    original_len = meta["length"] # 信源编码后的比特长度，信道编码前的比特长度
    if method == "CRC":
        payload = bits[:-8] if len(bits) >= 8 else np.zeros(0, dtype=np.uint8)
        recv_crc = bits_to_bytes(bits[-8:])[0] if len(bits) >= 8 else 0
        calc_crc = zlib.crc32(bits_to_bytes(payload)) & 0xFF
        return payload[:original_len], recv_crc == calc_crc
    if method == "汉明码":
        return hamming74_decode(bits, original_len), None
    return viterbi_decode(bits, original_len), None

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

"""实现卷积编码，使用一个简单的(2,1,2)卷积码，输入是原始比特流，输出是编码后的比特流。"""
def convolutional_encode(bits: np.ndarray) -> np.ndarray:
    state = [0, 0] # 2-bit状态寄存器，初始状态为00
    out: list[int] = []
    tail = np.zeros(2, dtype=np.uint8) # 尾比特，保持状态寄存器回到00状态
    for bit in np.r_[bits, tail]:
        u = int(bit) # 将numpy的uint8类型转换为Python的int类型，提高运算速度
        out1 = u ^ state[0] ^ state[1] # 输出比特1，生成多项式为(1,1,1)
        out2 = u ^ state[1]              # 输出比特2，生成多项式为(1,0,1)
        out.extend([out1, out2])
        state = [u, state[0]] # 更新状态寄存器，新的状态由当前输入和之前的状态决定
    return np.array(out, dtype=np.uint8)

"""实现卷积码的维特比译码，输入是接收的比特流和原始比特长度，输出是解码后的比特流。"""
def viterbi_decode(bits: np.ndarray, original_len: int) -> np.ndarray:
    if len(bits) % 2:
        bits = bits[:-1]
    transitions: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for state in range(4):
        m1 = (state >> 1) & 1
        m2 = state & 1
        for u in (0, 1):
            next_state = ((u << 1) | m1) & 0b11
            transitions[(state, u)] = (next_state, (u ^ m1 ^ m2, u ^ m2))
    steps = len(bits) // 2
    prev_state = np.full((steps, 4), -1, dtype=np.int8)
    decision = np.zeros((steps, 4), dtype=np.uint8)
    metrics = np.full(4, np.inf)
    metrics[0] = 0.0

    for step in range(steps):
        symbol = bits[step * 2 : step * 2 + 2]
        new_metrics = np.full(4, np.inf)
        for state in range(4):
            if not np.isfinite(metrics[state]):
                continue
            for u in (0, 1):
                next_state, expected = transitions[(state, u)]
                distance = int(expected[0] != symbol[0]) + int(expected[1] != symbol[1])
                score = metrics[state] + distance
                if score < new_metrics[next_state]:
                    new_metrics[next_state] = score
                    prev_state[step, next_state] = state
                    decision[step, next_state] = u
        metrics = new_metrics

    state = 0 if np.isfinite(metrics[0]) else int(np.argmin(metrics))
    decoded = np.zeros(steps, dtype=np.uint8)
    for step in range(steps - 1, -1, -1):
        decoded[step] = decision[step, state]
        state = prev_state[step, state]
        if state < 0:
            state = 0
    return decoded[:original_len]

"""生成星座点，输入是调制方式和阶数，输出是对应的复数星座点数组，并进行归一化使得平均功率为1。支持MASK、MPSK和MQAM三种调制方式。"""
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
    return (points / np.sqrt(np.mean(np.abs(points) ** 2, dtype=np.float64))).astype(np.complex64)  # 对星座点进行归一化，使得平均功率为1

"""生成根升余弦脉冲成型滤波器的系数，输入是滚降系数、符号跨度和每符号采样数，输出是滤波器的时域系数数组。"""
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


def modulate(bits: np.ndarray, modulation: str, order: int, roll_off: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = int(round(math.log2(order)))   # 计算每符号包含几个比特
    """比特对齐"""
    if len(bits) % width:
        bits = np.r_[bits, np.zeros((-len(bits)) % width, dtype=np.uint8)]
    indices = np.array(bits_to_ints(bits, width), dtype=np.uint32) # 将比特流转换为符号索引
    symbols = constellation(modulation, order)[indices].astype(np.complex64) # 根据调制方式和阶数获取星座点，并映射符号索引到对应的复数星座点
    upsampled = np.zeros(len(symbols) * SPS, dtype=np.complex64) # 用于存储上采样后的时域信号
    upsampled[::SPS] = symbols # 将符号映射到上采样后的时域信号中，符号之间插入SPS-1个零
    pulse = root_raised_cosine(roll_off) # 生成根升余弦脉冲成型滤波器的系数
    signal = np.convolve(upsampled, pulse.astype(np.float32), mode="full").astype(np.complex64)
    return signal, symbols.astype(np.complex64), pulse.astype(np.float32)

"""将成型滤波后的时域信号通过信道，添加瑞利或莱斯衰落和AWGN噪声，输入是成型滤波后的时域信号、调制后的符号、信道模型名称、SNR和K因子，输出是经过信道后的接收信号和衰落系数。"""
def apply_channel(tx_signal: np.ndarray, tx_symbols: np.ndarray, channel_name: str, snr_db: float, k_factor: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng() # 创建一个新的随机数生成器实例，使用默认的算法和随机种子，避免全局状态的干扰
    fading = np.ones(len(tx_symbols), dtype=np.complex64)
    if channel_name == "瑞利衰落":
        fading = ((rng.normal(size=len(tx_symbols)) + 1j * rng.normal(size=len(tx_symbols))) / np.sqrt(2)).astype(np.complex64)
    elif channel_name == "莱斯衰落":
        los = math.sqrt(k_factor / (k_factor + 1)) 
        scatter = math.sqrt(1 / (k_factor + 1))
        fading = (
            los + scatter * (rng.normal(size=len(tx_symbols)) + 1j * rng.normal(size=len(tx_symbols))) / np.sqrt(2)
        ).astype(np.complex64)
    fading_samples = np.repeat(fading, SPS)  # 假设一个符号周期内的衰落系数保持不变（即平坦衰落）
    if len(fading_samples) < len(tx_signal):
        fading_samples = np.r_[fading_samples, np.repeat(fading[-1], len(tx_signal) - len(fading_samples))]
    rx = tx_signal * fading_samples[: len(tx_signal)] # 添加乘性干扰
    power = np.mean(np.abs(rx) ** 2, dtype=np.float64) + 1e-12 # 算出接收信号的平均功率（即每个采样点的功率），添加一个小常数以避免除零错误
    noise_power = power * SPS / (10 ** (snr_db / 10))
    noise = rng.normal(scale=np.sqrt(noise_power / 2), size=len(rx)) + 1j * rng.normal(
        scale=np.sqrt(noise_power / 2), size=len(rx)
    )  # 添加复数高斯白噪声(加性干扰)
    return (rx + noise).astype(np.complex64), fading.astype(np.complex64)

def demodulate(
        rx_signal: np.ndarray,
        pulse:np.ndarray,
        fading: np.ndarray,
        modulation: str,
        order: int,
        expected_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matched = np.convolve(rx_signal, pulse[::-1].conjugate(), mode="full").astype(np.complex64) # 匹配滤波，使用与根升余弦滤波器系数共轭并反转的脉冲进行卷积
    # 由于卷积引入的延迟，采样点从pulse长度-1开始，每隔SPS个采样点取一个样本
    sample_start = len(pulse) - 1
    sample_points = sample_start + np.arange(len(fading)) * SPS
    sampled = matched[sample_points]  # 得到采样值
    equalized = (sampled / np.where(np.abs(fading) < 1e-8, 1.0 + 0j, fading)).astype(np.complex64) # 进行零强迫信道均衡，抵消衰落影响
    """进行符号判决"""
    points = constellation(modulation, order)
    distances = np.abs(equalized[:, None] - points[None, :])
    indices = np.argmin(distances, axis=1).astype(np.int32)
    bits = ints_to_bits(indices.tolist(), int(round(math.log2(order))))[:expected_bits]
    return matched, sampled, points[indices], bits

"""将信道解码后的比特流进行信源解码，输入是信道解码后的比特流、信源编码的元信息和信源编码方法名称，输出是恢复的字节数据。"""
def source_decode(bits: np.ndarray, meta: dict, method: str) -> bytes:
    if method == "算术编码":
        return ArithmeticCoder.decode(bits, meta)
    reverse = {code: symbol for symbol, code in meta["codes"].items()} # 构建反向编码表，键是二进制字符串编码，值是对应的符号
    current = ""
    out = bytearray()
    for bit in bits.tolist():
        current += str(int(bit))
        if current in reverse:
            out.append(reverse[current])
            current = ""
            if len(out) == meta["length"]:
                break
    return bytes(out)

"""根据输入的类型和恢复的字节数据，进行相应的解析和转换，输出恢复的文本内容、图像像素矩阵、音频采样值、采样率以及WAV格式的音频数据（如果适用）。如果恢复失败，则返回相应的错误信息和空值。"""
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
) -> SimulationSession:
    return SimulationSession(
        SimulationConfig(kind, text, path, source_method, channel_method, modulation, order, channel_name, snr_db, k_factor, roll_off)
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
) -> SimulationResult:
    return create_session(kind, text, path, source_method, channel_method, modulation, order, channel_name, snr_db, k_factor, roll_off).run_all()

"""遗留代码，暂时没用到"""
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