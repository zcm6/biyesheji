from __future__ import annotations

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
    media_name: str
    snr_db: float
    repeat_index: int
    target_bits: np.ndarray
    model_values: np.ndarray
    viterbi_ber: float


@dataclass(frozen=True)
class MediaValidationReport:
    ai_avg_ber: float
    viterbi_avg_ber: float
    ai_media_ber: dict[str, float]
    viterbi_media_ber: dict[str, float]
    case_count: int

    @property
    def accepted(self) -> bool:
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
    """Precompute fixed media channel outputs for model validation."""
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
    return {
        "ai_avg_ber": report.ai_avg_ber,
        "viterbi_avg_ber": report.viterbi_avg_ber,
        "accepted": report.accepted,
        "ai_media_ber": report.ai_media_ber,
        "viterbi_media_ber": report.viterbi_media_ber,
        "case_count": report.case_count,
    }


def format_report(step: int | None, report: MediaValidationReport) -> str:
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
