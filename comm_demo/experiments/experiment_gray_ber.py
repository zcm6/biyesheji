from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, rcParams
import matplotlib.pyplot as plt
import numpy as np

from comm_demo.core.session import run_pipeline


IMAGE_PATH = Path("test_data") / "demo_image.png"
RESULT_DIR = Path("experiment_results")
SNR_DB = 4.0
REPEATS = 3

MODULATIONS = [
    ("16ASK", "MASK"),
    ("16PSK", "MPSK"),
    ("16QAM", "MQAM"),
]


def configure_chinese_font() -> None:
    preferred_fonts = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            rcParams["font.sans-serif"] = [font_name]
            rcParams["font.family"] = "sans-serif"
            rcParams["axes.unicode_minus"] = False
            return


def run_once(modulation: str, gray_ok: bool) -> float:
    result = run_pipeline(
        kind="图像",
        text="",
        path=str(IMAGE_PATH),
        source_method="哈夫曼编码",
        channel_method="卷积码",
        modulation=modulation,
        order=16,
        channel_name="AWGN",
        snr_db=SNR_DB,
        k_factor=3.0,
        roll_off=0.35,
        gray_ok=gray_ok,
    )
    return float(result.ber)


def main() -> None:
    configure_chinese_font()

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    RESULT_DIR.mkdir(exist_ok=True)

    rows: list[tuple[str, str, float, float, list[float]]] = []
    means = {"natural": [], "gray": []}

    for display_name, modulation in MODULATIONS:
        for gray_ok, key, label in [(False, "natural", "Natural"), (True, "gray", "Gray")]:
            values = [run_once(modulation, gray_ok) for _ in range(REPEATS)]
            mean = float(np.mean(values))
            std = float(np.std(values))
            rows.append((display_name, label, mean, std, values))
            means[key].append(mean)
            print(
                f"{display_name:5s} {label:7s} "
                f"BER mean={mean:.6f}, std={std:.6f}, runs={values}"
            )

    csv_path = RESULT_DIR / "gray_ber_comparison.csv"
    improvements = [
        (natural - gray) / natural * 100 if natural > 0 else 0.0
        for natural, gray in zip(means["natural"], means["gray"])
    ]
    with csv_path.open("w", encoding="utf-8") as file:
        file.write("modulation,mapping,ber_mean,ber_std,improvement_percent,runs\n")
        for display_name, label, mean, std, values in rows:
            run_text = ";".join(f"{value:.8f}" for value in values)
            improvement = ""
            if label == "Gray":
                modulation_index = [name for name, _ in MODULATIONS].index(display_name)
                improvement = f"{improvements[modulation_index]:.2f}"
            file.write(f"{display_name},{label},{mean:.8f},{std:.8f},{improvement},{run_text}\n")

    x = np.arange(len(MODULATIONS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    bars1 = ax.bar(x - width / 2, means["natural"], width, label="自然映射", color="#6baed6", zorder=2)
    bars2 = ax.bar(x + width / 2, means["gray"], width, label="格雷码", color="#fdae6b", zorder=2)

    #ax.set_title("4 dB AWGN下自然映射和格雷码的BER对比图")
    ax.set_xlabel("调制方式")
    ax.set_ylabel("BER")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in MODULATIONS])
    ax.grid(axis="y", alpha=0.3, zorder=0)

    ax2 = ax.twinx()
    ax2.patch.set_visible(False)
    left_top = max(max(means["natural"]), max(means["gray"])) * 1.12
    right_top = max(improvements) * 1.25 if improvements else 1
    ax.set_ylim(0, left_top)
    ax2.set_ylim(0, right_top)
    improvement_line_y = np.array(improvements) / right_top * left_top
    line = ax.plot(
        x,
        improvement_line_y,
        color="#d62728",
        linestyle="--",
        marker="o",
        linewidth=2,
        label="BER下降比例",
        zorder=4,
    )
    ax2.set_ylabel("BER下降比例(%)")

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=6,
            )
    for index, value in enumerate(improvements):
        ax.annotate(
            f"{value:.1f}%",
            xy=(x[index], improvement_line_y[index]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#d62728",
            zorder=6,
        )

    handles1, labels1 = ax.get_legend_handles_labels()
    ax.legend(
        handles1,
        labels1,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=1,
        framealpha=0.85,
    )

    fig.tight_layout()
    png_path = RESULT_DIR / "gray_ber_comparison.png"
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved chart: {png_path}")
    print(f"Saved data:  {csv_path}")


if __name__ == "__main__":
    main()
