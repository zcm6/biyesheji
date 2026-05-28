from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .bits import bytes_to_bits
from .channel import apply_channel
from .channel_coding import channel_decode, channel_encode
from .constants import DEFAULT_ROLL_OFF, STAGE_NAMES
from .models import SimulationConfig, SimulationResult, SourceArtifact
from .modulation import demodulate, modulate
from .source import prepare_source, restore_output
from .source_coding import source_decode, source_encode


@dataclass
class SimulationSession:
    """管理一次通信链路仿真的分步执行状态。

    Attributes:
        config: 本次仿真的参数配置。
        stage_index: 当前执行到的阶段索引。
        history: 已完成阶段的文字记录。
        source: 信源预处理结果。
        source_bits: 原始信源比特流。
        source_coded_bits: 信源编码后的比特流。
        source_meta: 信源解码所需的元数据。
        channel_coded_bits: 信道编码后的比特流。
        channel_meta: 信道解码所需的元数据。
        tx_symbols: 调制后的发送星座符号。
        tx_signal: 脉冲成形后的发送信号。
        pulse: 根升余弦脉冲成形滤波器。
        rx_signal: 经过信道后的接收信号。
        fading_symbols: 符号级信道衰落系数。
        matched_signal: 匹配滤波后的信号。
        sampled_symbols: 均衡后的采样符号。
        detected_symbols: 判决后的星座符号。
        rx_channel_bits: 解调恢复出的信道编码比特流。
        decoded_source_bits: 信道解码后的信源编码比特流。
        restored_bytes: 信源解码恢复出的字节数据。
        stage_times: 各阶段的执行耗时。
    """

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
        """判断仿真流程是否已经执行完成。

        Returns:
            如果所有阶段均已完成，则返回 ``True``；否则返回 ``False``。
        """
        return self.stage_index >= len(STAGE_NAMES)

    def next_stage(self) -> str:
        """返回当前待执行的仿真阶段名称。

        Returns:
            当前阶段名称；若流程已完成，则返回 ``"已完成"``。
        """
        return "已完成" if self.is_finished() else STAGE_NAMES[self.stage_index]

    def step(self) -> str:
        """执行通信链路中的下一个仿真阶段。

        Returns:
            当前阶段完成后的历史记录文本。
        """
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
                self.pulse,
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
                self.rx_channel_bits,
                self.channel_meta,
                self.config.channel_method,
                ai_decoder=self.config.ai_decoder,
                modulation=self.config.modulation,
                order=self.config.order,
                channel_name=self.config.channel_name,
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
        print(f"[core] {stage_name}: {elapsed * 1000:.2f} ms")
        self.stage_index += 1
        self.history.append(f"{self.stage_index}. {message}")
        return self.history[-1]

    def run_all(self) -> SimulationResult:
        """连续执行所有剩余仿真阶段并生成完整结果。

        Returns:
            包含全链路中间数据和恢复结果的 SimulationResult 对象。
        """
        while not self.is_finished():
            self.step()
        return self.build_result()

    def build_result(self) -> SimulationResult:
        """根据当前会话状态构建完整仿真结果对象。

        Returns:
            汇总信源、编码、调制、信道、解调和恢复结果的 SimulationResult。
        """
        restored_text, restored_image, restored_audio_samples, restored_audio_rate, restored_audio_wav = restore_output(
            self.source.kind, self.restored_bytes or b"", self.source.media_meta
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
    ai_decoder: bool = False,
) -> SimulationSession:
    """根据用户配置创建分步仿真会话。

    Args:
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

    Returns:
        初始化后的 SimulationSession 对象。
    """
    return SimulationSession(
        SimulationConfig(
            kind,
            text,
            path,
            source_method,
            channel_method,
            modulation,
            order,
            channel_name,
            snr_db,
            k_factor,
            roll_off,
            gray_ok,
            ai_decoder,
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
    ai_decoder: bool = False,
) -> SimulationResult:
    """按给定参数直接运行完整通信链路仿真。

    Args:
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

    Returns:
        完整仿真结果对象。
    """
    return create_session(
        kind,
        text,
        path,
        source_method,
        channel_method,
        modulation,
        order,
        channel_name,
        snr_db,
        k_factor,
        roll_off,
        gray_ok,
        ai_decoder,
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
    """运行不含信源和信道编码的原始调制解调链路。

    Args:
        bits: 待调制的一维比特数组。
        modulation: 调制方式。
        order: 调制阶数。
        channel_name: 信道模型名称。
        snr_db: 信噪比，单位为 dB。
        k_factor: 莱斯衰落信道的 K 因子。
        roll_off: 根升余弦滤波器滚降系数。

    Returns:
        一个三元组，依次为解调后的比特流、均衡后的采样符号和匹配滤波信号。
    """
    tx_signal, tx_symbols, pulse = modulate(bits.astype(np.uint8), modulation, order, roll_off)
    rx_signal, fading_symbols = apply_channel(tx_signal, tx_symbols, channel_name, snr_db, k_factor, pulse)
    matched_signal, sampled_symbols, _, detected_bits = demodulate(
        rx_signal, pulse, fading_symbols, modulation, order, len(bits)
    )
    return detected_bits, sampled_symbols, matched_signal
