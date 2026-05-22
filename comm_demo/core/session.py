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
        while not self.is_finished():
            self.step()
        return self.build_result()

    def build_result(self) -> SimulationResult:
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
    tx_signal, tx_symbols, pulse = modulate(bits.astype(np.uint8), modulation, order, roll_off)
    rx_signal, fading_symbols = apply_channel(tx_signal, tx_symbols, channel_name, snr_db, k_factor, pulse)
    matched_signal, sampled_symbols, _, detected_bits = demodulate(
        rx_signal, pulse, fading_symbols, modulation, order, len(bits)
    )
    return detected_bits, sampled_symbols, matched_signal
