from __future__ import annotations

"""AI-BiGRU 卷积码译码器的固定媒体 BER 验证工具。

该模块使用固定图像和语音样本构造接收端测试序列，对比 AI-BiGRU
译码与传统 Viterbi 译码的平均 BER，并为训练过程提供模型接受依据。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comm_demo.core.channel import apply_channel
from comm_demo.core.channel_coding import convolutional_encode, viterbi_decode
from comm_demo.core.modulation import demodulate, modulate
from comm_demo.core.source import prepare_source
from comm_demo.core.source_coding import source_encode

try:
    from .data import _soft_bits_from_equalized, bits_to_observations
    from .inference import load_model, select_model_path
    from .model import require_torch, torch
except ImportError:
    from data import _soft_bits_from_equalized, bits_to_observations
    from inference import load_model, select_model_path
    from model import require_torch, torch


DEFAULT_MEDIA = (
    ("image", "图像", PROJECT_ROOT / "test_data" / "demo_image.png"),
    ("audio", "语音", PROJECT_ROOT / "test_data" / "demo_speech_zh.wav"),
)


@dataclass(frozen=True)
class MediaValidationCase:
    """保存一个固定媒体链路验证样本。

    Attributes:
        media_name: 媒体样本名称，例如图像或语音。
        snr_db: 当前样本使用的信噪比，单位为 dB。
        repeat_index: 当前信噪比下的重复实验编号。
        target_bits: 原始目标比特流。
        model_values: 输入 AI-BiGRU 模型的接收端观测序列。
        viterbi_ber: 同一接收序列经过 Viterbi 译码得到的 BER。
    """

    media_name: str
    snr_db: float
    repeat_index: int
    target_bits: np.ndarray
    model_values: np.ndarray
    viterbi_ber: float


@dataclass(frozen=True)
class MediaValidationReport:
    """保存固定媒体 BER 验证的汇总结果。

    Attributes:
        ai_avg_ber: AI-BiGRU 在所有媒体上的平均 BER。
        viterbi_avg_ber: Viterbi 译码在所有媒体上的平均 BER。
        ai_media_ber: AI-BiGRU 按媒体类型统计的平均 BER。
        viterbi_media_ber: Viterbi 译码按媒体类型统计的平均 BER。
        case_count: 参与统计的验证样本数量。
    """

    ai_avg_ber: float
    viterbi_avg_ber: float
    ai_media_ber: dict[str, float]
    viterbi_media_ber: dict[str, float]
    case_count: int

    @property
    def accepted(self) -> bool:
        """判断 AI-BiGRU 验证结果是否满足保存条件。

        Returns:
            当 AI-BiGRU 平均 BER 不高于 Viterbi 平均 BER 时返回 ``True``。
        """
        return self.ai_avg_ber <= self.viterbi_avg_ber


def _channel_values(
    coded_bits: np.ndarray,
    modulation: str,
    order: int,
    channel_name: str,
    snr_db: float,
    k_factor: float,
    roll_off: float,
    gray_ok: bool,
    input_mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """通过项目主通信链路生成接收端模型输入和硬判决比特。

    Args:
        coded_bits: 卷积编码后的发送比特流。
        modulation: 调制方式。
        order: 调制阶数。
        channel_name: 信道模型名称。
        snr_db: 信噪比，单位为 dB。
        k_factor: 莱斯衰落信道的 K 因子。
        roll_off: 根升余弦滤波器滚降系数。
        gray_ok: 是否启用格雷映射。
        input_mode: 模型输入模式，支持 ``"hard"`` 和 ``"soft"``。
        rng: NumPy 随机数生成器。

    Returns:
        一个二元组，包含输入模型的接收端观测值，以及解调得到的硬判决比特。
    """
    tx_signal, tx_symbols, pulse = modulate(coded_bits, modulation, order, roll_off, gray_ok)
    rx_signal, fading_symbols = apply_channel(
        tx_signal,
        tx_symbols,
        channel_name,
        snr_db,
        k_factor,
        pulse,
        rng=rng,
    )
    _, equalized_symbols, _, hard_bits = demodulate(
        rx_signal,
        pulse,
        fading_symbols,
        modulation,
        order,
        len(coded_bits),
        gray_ok,
    )
    if input_mode == "soft":
        model_values = _soft_bits_from_equalized(
            equalized_symbols,
            len(coded_bits),
            modulation,
            order,
            gray_ok,
        )
    else:
        model_values = hard_bits
    return model_values.astype(np.float32 if input_mode == "soft" else np.uint8, copy=False), hard_bits


def build_validation_cases(
    modulation: str,
    order: int,
    channel_name: str = "AWGN",
    snrs: tuple[float, ...] = (0, 2, 4, 6, 8, 10, 12),
    repeats: int = 3,
    input_mode: str = "hard",
    source_method: str = "哈夫曼编码",
    k_factor: float = 3.0,
    roll_off: float = 0.35,
    gray_ok: bool = True,
    seed: int = 20260521,
) -> list[MediaValidationCase]:
    """预生成固定媒体链路验证样本。

    Args:
        modulation: 验证使用的调制方式。
        order: 验证使用的调制阶数。
        channel_name: 验证使用的信道模型名称。
        snrs: 参与验证的信噪比集合。
        repeats: 每个信噪比下的重复次数。
        input_mode: 模型输入模式，支持 ``"hard"`` 和 ``"soft"``。
        source_method: 固定媒体样本使用的信源编码方式。
        k_factor: 莱斯衰落信道的 K 因子。
        roll_off: 根升余弦滤波器滚降系数。
        gray_ok: 是否启用格雷映射。
        seed: 验证样本生成使用的随机种子。

    Returns:
        MediaValidationCase 列表，包含固定媒体在不同 SNR 和重复次数下的
        接收端观测、目标比特和 Viterbi BER。
    """
    rng = np.random.default_rng(seed)
    cases: list[MediaValidationCase] = []
    media_bits: dict[str, np.ndarray] = {}

    for media_name, kind, path in DEFAULT_MEDIA:
        source = prepare_source(kind, "", str(path))
        source_bits, _ = source_encode(source.raw_bytes, source_method)
        media_bits[media_name] = source_bits.astype(np.uint8, copy=False)

    for media_name, source_bits in media_bits.items():
        coded_bits = convolutional_encode(source_bits)
        for snr_db in snrs:
            for repeat_index in range(repeats):
                model_values, hard_bits = _channel_values(
                    coded_bits,
                    modulation,
                    order,
                    channel_name,
                    float(snr_db),
                    k_factor,
                    roll_off,
                    gray_ok,
                    input_mode,
                    rng,
                )
                viterbi_bits = viterbi_decode(hard_bits, len(source_bits))
                viterbi_ber = float(np.mean(viterbi_bits != source_bits)) if len(source_bits) else 0.0
                cases.append(
                    MediaValidationCase(
                        media_name=media_name,
                        snr_db=float(snr_db),
                        repeat_index=repeat_index,
                        target_bits=source_bits,
                        model_values=model_values,
                        viterbi_ber=viterbi_ber,
                    )
                )
    return cases


def decode_model_values(
    model,
    received_values: np.ndarray,
    original_len: int,
    input_mode: str,
    device: str,
    window_steps: int = 8192,
) -> np.ndarray:
    """使用 AI-BiGRU 模型对接收端观测序列进行分块译码。

    Args:
        model: 已加载的 BiGRU 译码模型。
        received_values: 接收端硬判决比特或软信息序列。
        original_len: 需要恢复的原始信息比特长度。
        input_mode: 模型输入模式，支持 ``"hard"`` 和 ``"soft"``。
        device: PyTorch 推理设备。
        window_steps: 单次送入模型的最大时间步数。

    Returns:
        uint8 类型的一维数组，包含 AI-BiGRU 译码恢复出的信息比特。
    """
    observations = bits_to_observations(received_values, input_mode=input_mode)
    if observations.size == 0:
        return np.zeros(0, dtype=np.uint8)

    decoded = np.empty(min(original_len, len(observations)), dtype=np.uint8)
    with torch.no_grad():
        for start in range(0, decoded.size, window_steps):
            output_stop = min(start + window_steps, decoded.size)
            observation_stop = min(output_stop + 2, len(observations))
            x = torch.from_numpy(observations[start:observation_stop][None, :, :]).to(device)
            logits = model(x)[0, : output_stop - start]
            decoded[start:output_stop] = (torch.sigmoid(logits) >= 0.5).to(torch.uint8).cpu().numpy()
    return decoded[:original_len]


def evaluate_cases(
    model,
    cases: list[MediaValidationCase],
    input_mode: str,
    device: str,
    window_steps: int = 8192,
) -> MediaValidationReport:
    """评估 AI-BiGRU 模型在固定媒体样本上的 BER 表现。

    Args:
        model: 待评估的 BiGRU 译码模型。
        cases: 预生成的固定媒体验证样本列表。
        input_mode: 模型输入模式，支持 ``"hard"`` 和 ``"soft"``。
        device: PyTorch 推理设备。
        window_steps: 单次送入模型的最大时间步数。

    Returns:
        汇总 AI-BiGRU 和 Viterbi 平均 BER 的 MediaValidationReport。
    """
    was_training = model.training
    model.eval()
    ai_by_media: dict[str, list[float]] = {}
    viterbi_by_media: dict[str, list[float]] = {}

    for case in cases:
        ai_bits = decode_model_values(
            model,
            case.model_values,
            len(case.target_bits),
            input_mode,
            device,
            window_steps=window_steps,
        )
        ai_ber = float(np.mean(ai_bits != case.target_bits)) if len(case.target_bits) else 0.0
        ai_by_media.setdefault(case.media_name, []).append(ai_ber)
        viterbi_by_media.setdefault(case.media_name, []).append(case.viterbi_ber)

    if was_training:
        model.train()

    ai_media = {name: float(np.mean(values)) for name, values in ai_by_media.items()}
    viterbi_media = {name: float(np.mean(values)) for name, values in viterbi_by_media.items()}
    return MediaValidationReport(
        ai_avg_ber=float(np.mean(list(ai_media.values()))),
        viterbi_avg_ber=float(np.mean(list(viterbi_media.values()))),
        ai_media_ber=ai_media,
        viterbi_media_ber=viterbi_media,
        case_count=len(cases),
    )


def report_to_dict(report: MediaValidationReport) -> dict:
    """将媒体 BER 验证报告转换为可保存的字典。

    Args:
        report: 媒体 BER 验证报告对象。

    Returns:
        包含平均 BER、分媒体 BER、样本数量和接受标志的字典。
    """
    return {
        "ai_avg_ber": report.ai_avg_ber,
        "viterbi_avg_ber": report.viterbi_avg_ber,
        "accepted": report.accepted,
        "ai_media_ber": report.ai_media_ber,
        "viterbi_media_ber": report.viterbi_media_ber,
        "case_count": report.case_count,
    }


def format_report(step: int | None, report: MediaValidationReport) -> str:
    """将媒体 BER 验证报告格式化为日志字符串。

    Args:
        step: 当前训练步数；独立验证时可为 ``None``。
        report: 媒体 BER 验证报告对象。

    Returns:
        适合打印到控制台的一行验证结果字符串。
    """
    prefix = f"step={step:5d} " if step is not None else ""
    return (
        f"{prefix}media_ai_avg_ber={report.ai_avg_ber:.8f} "
        f"media_viterbi_avg_ber={report.viterbi_avg_ber:.8f} "
        f"image_ai={report.ai_media_ber.get('image', float('nan')):.8f} "
        f"image_viterbi={report.viterbi_media_ber.get('image', float('nan')):.8f} "
        f"audio_ai={report.ai_media_ber.get('audio', float('nan')):.8f} "
        f"audio_viterbi={report.viterbi_media_ber.get('audio', float('nan')):.8f} "
        f"accepted={report.accepted}"
    )


def main() -> None:
    """解析命令行参数并对指定模型执行固定媒体 BER 验证。"""
    require_torch()
    parser = argparse.ArgumentParser(description="Validate a BiGRU decoder on fixed image and audio media cases.")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--modulation", choices=["MASK", "MPSK", "MQAM"], default="MQAM")
    parser.add_argument("--order", type=int, default=16)
    parser.add_argument("--channel-name", default="AWGN")
    parser.add_argument("--snrs", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--source-method", default="哈夫曼编码")
    parser.add_argument("--k-factor", type=float, default=3.0)
    parser.add_argument("--roll-off", type=float, default=0.35)
    parser.add_argument("--gray-ok", action="store_true", default=True)
    parser.add_argument("--no-gray", dest="gray_ok", action="store_false")
    parser.add_argument("--validation-seed", type=int, default=20260521)
    parser.add_argument("--window-steps", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path = args.model or select_model_path(args.modulation, args.order, args.channel_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    cases = build_validation_cases(
        modulation=args.modulation,
        order=args.order,
        channel_name=args.channel_name,
        snrs=tuple(args.snrs),
        repeats=args.repeats,
        input_mode=args.input_mode,
        source_method=args.source_method,
        k_factor=args.k_factor,
        roll_off=args.roll_off,
        gray_ok=args.gray_ok,
        seed=args.validation_seed,
    )
    model = load_model(model_path, args.device)
    report = evaluate_cases(model, cases, args.input_mode, args.device, args.window_steps)
    print(f"model: {model_path}")
    print(f"cases: {len(cases)} snrs={args.snrs} repeats={args.repeats}")
    print(format_report(None, report))


if __name__ == "__main__":
    main()
