from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_ROLL_OFF


@dataclass
class SourceArtifact:
    """保存信源预处理后的统一数据对象。

    Attributes:
        kind: 信源类型，例如文本、图像或语音。
        raw_bytes: 送入后续通信链路的原始字节载荷。
        preview_text: 用于界面展示的信源预览文本。
        text_value: 文本信源的原始字符串内容。
        image_array: 图像信源的灰度像素数组。
        audio_samples: 语音信源的采样数组。
        sample_rate: 语音信源的采样率。
        audio_wav_bytes: 用于播放预览的 WAV 格式音频字节。
        media_meta: 图像或语音恢复所需的媒体元数据。
    """

    kind: str
    raw_bytes: bytes
    preview_text: str
    text_value: str = ""
    image_array: np.ndarray | None = None
    audio_samples: np.ndarray | None = None
    sample_rate: int = 0
    audio_wav_bytes: bytes | None = None
    media_meta: dict[str, int] | None = None


@dataclass
class SimulationConfig:
    """保存一次通信链路仿真的配置参数。

    Attributes:
        kind: 信源类型。
        text: 文本信源内容。
        path: 图像或语音文件路径。
        source_method: 信源编码方式。
        channel_method: 信道编码方式。
        modulation: 调制方式。
        order: 调制阶数。
        channel_name: 信道模型名称。
        snr_db: 信噪比，单位为 dB。
        k_factor: 莱斯衰落信道的 K 因子。
        roll_off: 根升余弦滤波器滚降系数。
        gray_ok: 是否启用格雷映射。
        ai_decoder: 是否启用 AI 卷积码译码器。
    """

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
    ai_decoder: bool = False


@dataclass
class SimulationResult:
    """保存完整通信链路仿真的结果与中间数据。

    Attributes:
        config: 本次仿真使用的配置。
        source: 预处理后的信源数据。
        source_bits: 原始信源比特流。
        source_coded_bits: 信源编码后的比特流。
        channel_coded_bits: 信道编码后的比特流。
        tx_symbols: 调制后的复数星座符号。
        tx_signal: 上采样并脉冲成形后的发送信号。
        pulse: 根升余弦脉冲成形滤波器系数。
        rx_signal: 经过信道和噪声后的接收信号。
        matched_signal: 匹配滤波后的接收信号。
        sampled_symbols: 均衡后的采样符号。
        detected_symbols: 判决后的星座符号。
        rx_channel_bits: 解调得到的信道编码比特流。
        decoded_source_bits: 信道解码后的信源编码比特流。
        restored_bytes: 信源解码恢复出的字节数据。
        restored_text: 恢复出的文本内容。
        restored_image: 恢复出的图像数组。
        restored_audio_samples: 恢复出的语音采样数组。
        restored_audio_rate: 恢复语音的采样率。
        restored_audio_wav_bytes: 恢复语音的 WAV 格式字节。
        ber: 接收端恢复结果相对发送端参考比特的误比特率。
        history: 分步仿真过程记录。
        crc_ok: CRC 校验结果；非 CRC 信道编码时为 ``None``。
        gray_ok: 是否启用格雷映射的记录字段。
        stage_times: 各仿真阶段耗时。
    """

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
        """返回用于界面显示的仿真摘要文本。

        Returns:
            格式化后的仿真摘要字符串。
        """
        return self.summary_text()

    @property
    def src_bits(self) -> np.ndarray:
        """返回原始信源比特流。

        Returns:
            原始信源比特流数组。
        """
        return self.source_bits

    @property
    def preview(self) -> str:
        """返回恢复内容的预览文本。

        Returns:
            恢复文本或恢复状态说明。
        """
        return self.restored_text

    def summary_text(self) -> str:
        """生成格式化的仿真结果摘要。

        Returns:
            包含输入类型、编码方式、调制参数、信道条件和性能指标的
            多行摘要文本。
        """
        lines = [
            f"输入类型: {self.config.kind}",
            f"信源编码: {self.config.source_method}",
            f"信道编码: {self.config.channel_method}",
            f"AI译码: {'是' if self.config.ai_decoder else '否'}",
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

    def bit_views(self) -> dict[str, np.ndarray]:
        """返回用于比特流可视化的各阶段比特数据。

        Returns:
            以视图名称为键、比特数组为值的字典。
        """
        return {
            "原始比特流": self.source_bits,
            "信源编码后": self.source_coded_bits,
            "信道编码后": self.channel_coded_bits,
            "判决后比特流": self.rx_channel_bits,
            "信道解码后": self.decoded_source_bits,
        }


    def signal_views(self) -> dict[str, np.ndarray]:
        """返回用于信号可视化的各阶段波形数据。

        Returns:
            以视图名称为键、复基带信号数组为值的字典。
        """
        return {
            "调制后信号": self.tx_signal,
            "信道后信号": self.rx_signal,
            "匹配滤波后信号": self.matched_signal,
        }
