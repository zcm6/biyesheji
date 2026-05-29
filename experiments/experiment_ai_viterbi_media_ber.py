from __future__ import annotations

import contextlib
import csv
import io
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

from adjustText import adjust_text
from matplotlib import font_manager, rcParams
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai卷积码译码.inference import load_model, preferred_device, select_model_path
from ai卷积码译码.media_validation import build_validation_cases, decode_model_values

try:
    from .common import ExperimentOutput
except ImportError:
    from common import ExperimentOutput


RESULT_DIR = Path("experiment_results") / "ai_viterbi_media_ber"
SNR_VALUES = (0, 2, 4, 6, 8, 10)
REPEATS = 3
INPUT_MODE = "hard"
CHANNEL_NAME = "AWGN"
ORDER = 16
ROLL_OFF = 0.35
GRAY_OK = True
VALIDATION_SEED = 20260521
WINDOW_STEPS = 8192
ZERO_BER_FLOOR = 1e-6
Y_AXIS_FLOOR = ZERO_BER_FLOOR / 2.0
Y_AXIS_FLOOR_BY_DISPLAY_NAME = {
    "16ASK": 1e-3,
    "16PSK": 1e-5,
}
MODULATIONS = (
    ("16ASK", "MASK"),
    ("16PSK", "MPSK"),
    ("16QAM", "MQAM"),
)
DECODERS = ("Viterbi", "AI-BiGRU")


@dataclass(frozen=True)
class RawResult:
    display_name: str
    modulation: str
    snr_db: float
    media_name: str
    repeat_index: int
    decoder: str
    ber: float
    model_path: str


@dataclass(frozen=True)
class SummaryResult:
    display_name: str
    modulation: str
    snr_db: float
    decoder: str
    ber_mean: float
    ber_std: float
    case_count: int


def configure_chinese_font() -> None:
    preferred_fonts = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            rcParams["font.sans-serif"] = [font_name]
            rcParams["font.family"] = "sans-serif"
            rcParams["axes.unicode_minus"] = False
            return


def evaluate_modulation(display_name: str, modulation: str, device: str) -> list[RawResult]:
    model_path = select_model_path(modulation, ORDER, CHANNEL_NAME)
    model = load_model(model_path, device)
    cases = build_validation_cases(
        modulation=modulation,
        order=ORDER,
        channel_name=CHANNEL_NAME,
        snrs=tuple(float(value) for value in SNR_VALUES),
        repeats=REPEATS,
        input_mode=INPUT_MODE,
        roll_off=ROLL_OFF,
        gray_ok=GRAY_OK,
        seed=VALIDATION_SEED,
    )
    rows: list[RawResult] = []

    for case in cases:
        ai_bits = decode_model_values(
            model,
            case.model_values,
            len(case.target_bits),
            INPUT_MODE,
            device,
            window_steps=WINDOW_STEPS,
        )
        ai_ber = float(np.mean(ai_bits != case.target_bits)) if len(case.target_bits) else 0.0
        common = {
            "display_name": display_name,
            "modulation": modulation,
            "snr_db": case.snr_db,
            "media_name": case.media_name,
            "repeat_index": case.repeat_index + 1,
            "model_path": str(model_path),
        }
        rows.append(RawResult(decoder="Viterbi", ber=case.viterbi_ber, **common))
        rows.append(RawResult(decoder="AI-BiGRU", ber=ai_ber, **common))

    print(f"{display_name}: model={model_path.name} cases={len(cases)} device={device}")
    return rows


def summarize(raw_rows: list[RawResult]) -> list[SummaryResult]:
    grouped: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row.display_name, row.modulation, row.snr_db, row.decoder)].append(row.ber)

    rows: list[SummaryResult] = []
    for display_name, modulation in MODULATIONS:
        for snr_db in SNR_VALUES:
            for decoder in DECODERS:
                values = grouped[(display_name, modulation, float(snr_db), decoder)]
                rows.append(
                    SummaryResult(
                        display_name=display_name,
                        modulation=modulation,
                        snr_db=float(snr_db),
                        decoder=decoder,
                        ber_mean=float(np.mean(values)),
                        ber_std=float(np.std(values)),
                        case_count=len(values),
                    )
                )
    return rows


def write_raw_csv(rows: list[RawResult]) -> Path:
    csv_path = RESULT_DIR / "ai_viterbi_media_ber_raw.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "display_name",
                "modulation",
                "order",
                "channel_name",
                "gray_ok",
                "roll_off",
                "input_mode",
                "snr_db",
                "media",
                "repeat",
                "decoder",
                "ber",
                "model_path",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.display_name,
                    row.modulation,
                    ORDER,
                    CHANNEL_NAME,
                    GRAY_OK,
                    ROLL_OFF,
                    INPUT_MODE,
                    f"{row.snr_db:g}",
                    row.media_name,
                    row.repeat_index,
                    row.decoder,
                    f"{row.ber:.10f}",
                    row.model_path,
                ]
            )
    return csv_path


def write_summary_csv(rows: list[SummaryResult]) -> Path:
    csv_path = RESULT_DIR / "ai_viterbi_media_ber_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "display_name",
                "modulation",
                "order",
                "channel_name",
                "snr_db",
                "decoder",
                "ber_mean",
                "ber_std",
                "case_count",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.display_name,
                    row.modulation,
                    ORDER,
                    CHANNEL_NAME,
                    f"{row.snr_db:g}",
                    row.decoder,
                    f"{row.ber_mean:.10f}",
                    f"{row.ber_std:.10f}",
                    row.case_count,
                ]
            )
    return csv_path


def label_for_ber(ber: float) -> str:
    return f"{ber:.6f}" if 0 < ber < 0.00005 else f"{ber:.4f}"


def configure_log_axis(ax: plt.Axes, y_top: float, y_floor: float = Y_AXIS_FLOOR) -> None:
    ax.set_ylim(y_floor, y_top)
    tick_floor = max(y_floor, ZERO_BER_FLOOR)
    major_ticks = 10.0 ** np.arange(
        int(np.ceil(np.log10(tick_floor))),
        int(np.floor(np.log10(y_top))) + 1,
    )
    minor_ticks = []
    for exponent in range(
        int(np.ceil(np.log10(tick_floor))),
        int(np.floor(np.log10(y_top))) + 1,
    ):
        decade = 10.0 ** exponent
        minor_ticks.extend(decade * np.arange(2, 10))
    minor_ticks = [tick for tick in minor_ticks if tick_floor <= tick <= y_top]
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(True, which="both", alpha=0.3, zorder=0)


def draw_figure(rows: list[SummaryResult]) -> list[Path]:
    by_series = {
        (row.display_name, row.decoder): []
        for row in rows
    }
    for row in rows:
        by_series[(row.display_name, row.decoder)].append(row)

    max_ber = max(row.ber_mean for row in rows)
    y_top = max_ber * 2.0 if max_ber > 0 else 1.0
    styles = {
        "Viterbi": {"color": "#4c78a8", "marker": "o"},
        "AI-BiGRU": {"color": "#f58518", "marker": "s"},
    }

    png_paths: list[Path] = []
    for display_name, _ in MODULATIONS:
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
        texts = []
        for decoder in DECODERS:
            series = sorted(by_series[(display_name, decoder)], key=lambda row: row.snr_db)
            real_values = np.array([row.ber_mean for row in series])
            plot_values = np.maximum(real_values, ZERO_BER_FLOOR)
            ax.semilogy(
                SNR_VALUES,
                plot_values,
                linewidth=2,
                markersize=5,
                label=decoder,
                zorder=3,
                **styles[decoder],
            )
            for snr_db, ber, plot_ber in zip(SNR_VALUES, real_values, plot_values):
                texts.append(
                    ax.text(
                        snr_db,
                        plot_ber,
                        label_for_ber(float(ber)),
                        fontsize=6.5,
                        color=styles[decoder]["color"],
                        ha="center",
                        va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.35),
                        zorder=5,
                    )
                )

        ax.set_title(display_name)
        ax.set_xlabel("SNR/dB")
        ax.set_xticks(SNR_VALUES)
        configure_log_axis(ax, y_top, Y_AXIS_FLOOR_BY_DISPLAY_NAME.get(display_name, Y_AXIS_FLOOR))
        ax.set_ylabel("平均 BER")
        ax.legend(loc="upper right", framealpha=0.85, fontsize=8)
        with contextlib.redirect_stdout(io.StringIO()):
            adjust_text(
                texts,
                ax=ax,
                expand=(1.08, 1.2),
                force_text=(0.18, 0.36),
                force_static=(0.08, 0.16),
                only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.35, alpha=0.6),
            )

        fig.tight_layout()
        png_path = RESULT_DIR / f"ai_viterbi_media_ber_{display_name.lower()}_semilogy.png"
        fig.savefig(png_path)
        plt.close(fig)
        png_paths.append(png_path)
    return png_paths


def main() -> None:
    configure_chinese_font()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    device = preferred_device()
    raw_rows: list[RawResult] = []
    for display_name, modulation in MODULATIONS:
        raw_rows.extend(evaluate_modulation(display_name, modulation, device))

    summary_rows = summarize(raw_rows)
    raw_csv_path = write_raw_csv(raw_rows)
    summary_csv_path = write_summary_csv(summary_rows)
    png_paths = draw_figure(summary_rows)

    for row in summary_rows:
        print(
            f"{row.display_name:5s} {row.decoder:8s} SNR={row.snr_db:2.0f} dB "
            f"BER mean={row.ber_mean:.8f} std={row.ber_std:.8f} cases={row.case_count}"
        )
    for png_path in png_paths:
        print(f"Saved chart:   {png_path}")
    print(f"Saved raw:     {raw_csv_path}")
    print(f"Saved summary: {summary_csv_path}")


def run_experiment(progress_callback=None) -> ExperimentOutput:
    """运行 AI-BiGRU 与 Viterbi 媒体 BER 对比实验，并返回输出文件清单。"""
    if progress_callback is not None:
        progress_callback("正在运行 AI-BiGRU 与 Viterbi 媒体 BER 对比实验...")
    main()
    return ExperimentOutput(
        title="AI-BiGRU 与 Viterbi BER 对比",
        result_dir=RESULT_DIR,
        image_paths=[
            RESULT_DIR / "ai_viterbi_media_ber_16ask_semilogy.png",
            RESULT_DIR / "ai_viterbi_media_ber_16psk_semilogy.png",
            RESULT_DIR / "ai_viterbi_media_ber_16qam_semilogy.png",
        ],
        csv_paths=[
            RESULT_DIR / "ai_viterbi_media_ber_raw.csv",
            RESULT_DIR / "ai_viterbi_media_ber_summary.csv",
        ],
        summary="比较 16ASK、16PSK、16QAM 下 AI-BiGRU 与硬判决 Viterbi 的媒体 BER。",
    )


if __name__ == "__main__":
    main()
