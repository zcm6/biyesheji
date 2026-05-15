from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, rcParams
import matplotlib.pyplot as plt
import numpy as np

from comm_demo.core.session import run_pipeline


IMAGE_PATH = Path("test_data") / "demo_image.png"
RESULT_DIR = Path("experiment_results") / "channel_code_snr"
SNR_VALUES = [0, 2, 4, 6, 8, 10]
CHANNEL_CODES = ["CRC", "汉明码", "卷积码"]
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


def run_once(channel_method: str, snr_db: float) -> float:
    result = run_pipeline(
        kind="图像",
        text="",
        path=str(IMAGE_PATH),
        source_method="哈夫曼编码",
        channel_method=channel_method,
        modulation="MQAM",
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

    results: dict[str, list[float]] = {method: [] for method in CHANNEL_CODES}
    std_results: dict[str, list[float]] = {method: [] for method in CHANNEL_CODES}
    rows: list[tuple[str, int, float, float, list[float]]] = []

    for channel_method in CHANNEL_CODES:
        for snr_db in SNR_VALUES:
            values = [run_once(channel_method, snr_db) for _ in range(REPEATS)]
            mean = float(np.mean(values))
            std = float(np.std(values))
            results[channel_method].append(mean)
            std_results[channel_method].append(std)
            rows.append((channel_method, snr_db, mean, std, values))
            print(
                f"{channel_method:4s} SNR={snr_db:2d} dB "
                f"BER mean={mean:.8f}, std={std:.8f}, runs={values}"
            )

    csv_path = RESULT_DIR / "channel_code_snr_ber.csv"
    with csv_path.open("w", encoding="utf-8") as file:
        file.write("channel_method,snr_db,ber_mean,ber_std,runs\n")
        for channel_method, snr_db, mean, std, values in rows:
            run_text = ";".join(f"{value:.8f}" for value in values)
            file.write(f"{channel_method},{snr_db},{mean:.8f},{std:.8f},{run_text}\n")

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    styles = {
        "CRC": {"color": "#4c78a8", "marker": "o"},
        "汉明码": {"color": "#f58518", "marker": "s"},
        "卷积码": {"color": "#54a24b", "marker": "^"},
    }
    high_snr_label_offsets = {
        "CRC": 6,
        "汉明码": 16,
        "卷积码": 26,
    }

    for channel_method in CHANNEL_CODES:
        ax.plot(
            SNR_VALUES,
            results[channel_method],
            linewidth=2,
            markersize=5,
            label=channel_method,
            zorder=3,
            **styles[channel_method],
        )
        for snr_db, ber in zip(SNR_VALUES, results[channel_method]):
            y_offset = high_snr_label_offsets[channel_method] if snr_db >= 6 else 6
            ax.annotate(
                f"{ber:.4f}",
                xy=(snr_db, ber),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=styles[channel_method]["color"],
            )

    # ax.set_title("不同信道编码方式对16QAM系统BER的影响")
    ax.set_xlabel("SNR(dB)")
    ax.set_ylabel("BER")
    ax.set_xticks(SNR_VALUES)
    max_ber = max(max(values) for values in results.values())
    ax.set_ylim(-max_ber * 0.04 if max_ber > 0 else -0.04, max_ber * 1.12 if max_ber > 0 else 1)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()

    png_path = RESULT_DIR / "channel_code_snr_ber.png"
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved chart: {png_path}")
    print(f"Saved data:  {csv_path}")


if __name__ == "__main__":
    main()
