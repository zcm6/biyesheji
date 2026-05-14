from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, rcParams
import matplotlib.pyplot as plt
import numpy as np

from comm_demo.pipeline import run_pipeline


IMAGE_PATH = Path("test_data") / "demo_image.png"
RESULT_DIR = Path("experiment_results") / "modulation_snr"
SNR_VALUES = [0, 2, 4, 6, 8, 10, 12]
MODULATIONS = [
    ("16ASK", "MASK"),
    ("16PSK", "MPSK"),
    ("16QAM", "MQAM"),
]
REPEATS = 3


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
    high_snr_label_offsets = {
        "16QAM": -8,
        "16PSK": 8,
        "16ASK": 24,
    }

    for display_name, _ in MODULATIONS:
        ax.plot(
            SNR_VALUES,
            results[display_name],
            linewidth=2,
            markersize=5,
            label=display_name,
            zorder=3,
            **styles[display_name],
        )
        for snr_db, ber in zip(SNR_VALUES, results[display_name]):
            y_offset = high_snr_label_offsets[display_name] if snr_db >= 8 else 6
            vertical_align = "top" if snr_db >= 8 and display_name == "16QAM" else "bottom"
            ax.annotate(
                f"{ber:.4f}",
                xy=(snr_db, ber),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va=vertical_align,
                fontsize=7,
                color=styles[display_name]["color"],
            )

    #ax.set_title("图5-x 不同调制方式在AWGN信道下的BER性能对比")
    ax.set_xlabel("SNR/dB")
    ax.set_ylabel("BER")
    ax.set_xticks(SNR_VALUES)
    max_ber = max(max(values) for values in results.values())
    ax.set_ylim(-max_ber * 0.08 if max_ber > 0 else -0.08, max_ber * 1.12 if max_ber > 0 else 1)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()

    png_path = RESULT_DIR / "modulation_snr_ber.png"
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved chart: {png_path}")
    print(f"Saved data:  {csv_path}")


if __name__ == "__main__":
    main()
