from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import DEFAULT_ROLL_OFF

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
    media_meta: dict[str, int] | None = None


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
