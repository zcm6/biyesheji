from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, rcParams
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comm_demo.core.session import run_pipeline

try:
    from .common import ExperimentOutput
except ImportError:
    from common import ExperimentOutput


IMAGE_PATH = Path("test_data") / "demo_image.png"
RESULT_DIR = Path("experiment_results") / "modulation_semilogy"
SNR_VALUES = [0, 2, 4, 6, 8, 10, 12]
MODULATIONS = [
    ("16ASK", "MASK"),
    ("16PSK", "MPSK"),
    ("16QAM", "MQAM"),
]
REPEATS = 3
ZERO_BER_FLOOR = 1e-6
Y_AXIS_FLOOR = ZERO_BER_FLOOR / 2.0


def configure_chinese_font() -> None:
    preferred_fonts = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            rcParams["font.sans-serif"] = [font_name]
            rcParams["font.family"] = "sans-serif"
            rcParams["axes.unicode_minus"] = False
            return


def run_once(modulation: str, snr_db: float) -> float:
    result = run_pipeline(
        kind="图像",
        text="",
        path=str(IMAGE_PATH),
        source_method="哈夫曼编码",
        channel_method="卷积码",
        modulation=modulation,
        order=16,
        channel_name="AWGN",
        snr_db=snr_db,
        k_factor=3.0,
        roll_off=0.35,
        gray_ok=True,
    )
    return float(result.ber)


def main() -> None:
    configure_chinese_font()

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[float]] = {display_name: [] for display_name, _ in MODULATIONS}
    rows: list[tuple[str, int, float, float, list[float]]] = []

    for display_name, modulation in MODULATIONS:
        for snr_db in SNR_VALUES:
            values = [run_once(modulation, snr_db) for _ in range(REPEATS)]
            mean = float(np.mean(values))
            std = float(np.std(values))
            results[display_name].append(mean)
            rows.append((display_name, snr_db, mean, std, values))
            print(
                f"{display_name:5s} SNR={snr_db:2d} dB "
                f"BER mean={mean:.8f}, std={std:.8f}, runs={values}"
            )

    csv_path = RESULT_DIR / "modulation_snr_ber.csv"
    with csv_path.open("w", encoding="utf-8") as file:
        file.write("modulation,snr_db,ber_mean,ber_std,runs\n")
        for display_name, snr_db, mean, std, values in rows:
            run_text = ";".join(f"{value:.8f}" for value in values)
            file.write(f"{display_name},{snr_db},{mean:.8f},{std:.8f},{run_text}\n")

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    styles = {
        "16ASK": {"color": "#4c78a8", "marker": "o"},
        "16PSK": {"color": "#f58518", "marker": "s"},
        "16QAM": {"color": "#54a24b", "marker": "^"},
    }
    texts = []

    for display_name, _ in MODULATIONS:
        plot_values = np.maximum(results[display_name], ZERO_BER_FLOOR)
        ax.semilogy(
            SNR_VALUES,
            plot_values,
            linewidth=2,
            markersize=5,
            label=display_name,
            zorder=3,
            **styles[display_name],
        )
        for snr_db, ber, plot_ber in zip(SNR_VALUES, results[display_name], plot_values):
            ber_label = f"{ber:.6f}" if 0 < ber < 0.00005 else f"{ber:.4f}"
            texts.append(
                ax.text(
                    snr_db,
                    plot_ber,
                    ber_label,
                    fontsize=7,
                    color=styles[display_name]["color"],
                    ha="center",
                    va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.4),
                    zorder=5,
                )
            )

    #ax.set_title("图5-x 不同调制方式在AWGN信道下的BER性能对比")
    ax.set_xlabel("SNR/dB")
    ax.set_ylabel("BER")
    ax.set_xticks(SNR_VALUES)
    positive_bers = [ber for values in results.values() for ber in values if ber > 0]
    min_ber = min(positive_bers) if positive_bers else ZERO_BER_FLOOR
    max_ber = max(max(values) for values in results.values())
    y_top = max_ber * 2.0 if max_ber > 0 else 1
    ax.set_ylim(Y_AXIS_FLOOR, y_top)
    major_ticks = 10.0 ** np.arange(
        int(np.ceil(np.log10(ZERO_BER_FLOOR))),
        int(np.floor(np.log10(y_top))) + 1,
    )
    minor_ticks = []
    for exponent in range(
        int(np.ceil(np.log10(ZERO_BER_FLOOR))),
        int(np.floor(np.log10(y_top))) + 1,
    ):
        decade = 10.0 ** exponent
        minor_ticks.extend(decade * np.arange(2, 10))
    minor_ticks = [tick for tick in minor_ticks if ZERO_BER_FLOOR <= tick <= y_top]
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(True, which="both", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", framealpha=0.85)
    with contextlib.redirect_stdout(io.StringIO()):
        adjust_text(
            texts,
            ax=ax,
            expand=(1.1, 1.25),
            force_text=(0.2, 0.4),
            force_static=(0.1, 0.2),
            only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.4, alpha=0.6),
        )
    fig.tight_layout()

    png_path = RESULT_DIR / "modulation_snr_ber.png"
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved chart: {png_path}")
    print(f"Saved data:  {csv_path}")


def run_experiment(progress_callback=None) -> ExperimentOutput:
    """运行不同调制方式 BER 对比实验，并返回输出文件清单。"""
    if progress_callback is not None:
        progress_callback("正在运行不同调制方式 BER 对比实验...")
    main()
    return ExperimentOutput(
        title="调制方式 BER 对比",
        result_dir=RESULT_DIR,
        image_paths=[RESULT_DIR / "modulation_snr_ber.png"],
        csv_paths=[RESULT_DIR / "modulation_snr_ber.csv"],
        summary="比较 16ASK、16PSK 和 16QAM 在 AWGN 信道下的 BER 曲线。",
    )


if __name__ == "__main__":
    main()

