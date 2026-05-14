from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, rcParams
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from comm_demo.core.session import run_pipeline


IMAGE_PATH = Path("test_data") / "demo_image.png"
RESULT_DIR = Path("experiment_results") / "rolloff_16qam"
SNR_DB = 4.0
REPEATS = 3
ROLL_OFFS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


def configure_chinese_font() -> None:
    preferred_fonts = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            rcParams["font.sans-serif"] = [font_name]
            rcParams["font.family"] = "sans-serif"
            rcParams["axes.unicode_minus"] = False
            return


def run_once(roll_off: float):
    return run_pipeline(
        kind="图像",
        text="",
        path=str(IMAGE_PATH),
        source_method="哈夫曼编码",
        channel_method="卷积码",
        modulation="MQAM",
        order=16,
        channel_name="AWGN",
        snr_db=SNR_DB,
        k_factor=3.0,
        roll_off=roll_off,
        gray_ok=True,
    )


def save_restored_image(image: np.ndarray | None, path: Path) -> None:
    if image is None:
        return
    restored = np.asarray(image, dtype=np.uint8)
    Image.fromarray(restored, mode="L").save(path)


def main() -> None:
    configure_chinese_font()

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[float, float, float, list[float]]] = []
    mean_bers: list[float] = []
    std_bers: list[float] = []

    for roll_off in ROLL_OFFS:
        values: list[float] = []
        image_saved = False
        for repeat_index in range(REPEATS):
            result = run_once(roll_off)
            values.append(float(result.ber))
            if not image_saved and result.restored_image is not None:
                image_path = RESULT_DIR / f"restored_rolloff_{roll_off:g}.png"
                save_restored_image(result.restored_image, image_path)
                image_saved = True
        mean = float(np.mean(values))
        std = float(np.std(values))
        rows.append((roll_off, mean, std, values))
        mean_bers.append(mean)
        std_bers.append(std)
        print(f"roll_off={roll_off:g} BER mean={mean:.8f}, std={std:.8f}, runs={values}")

    csv_path = RESULT_DIR / "rolloff_16qam_ber.csv"
    with csv_path.open("w", encoding="utf-8") as file:
        file.write("roll_off,ber_mean,ber_std,runs\n")
        for roll_off, mean, std, values in rows:
            run_text = ";".join(f"{value:.8f}" for value in values)
            file.write(f"{roll_off:g},{mean:.8f},{std:.8f},{run_text}\n")

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.plot(
        ROLL_OFFS,
        mean_bers,
        color="#2f6f9f",
        marker="o",
        linewidth=2,
        label="BER",
        zorder=3,
    )
    ax.fill_between(
        ROLL_OFFS,
        np.array(mean_bers) - np.array(std_bers),
        np.array(mean_bers) + np.array(std_bers),
        color="#9ecae1",
        alpha=0.35,
        label="标准差范围",
        zorder=2,
    )
    for roll_off, ber in zip(ROLL_OFFS, mean_bers):
        ax.annotate(
            f"{ber:.4f}",
            xy=(roll_off, ber),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("滚降系数对16QAM系统BER的影响")
    ax.set_xlabel("滚降系数")
    ax.set_ylabel("BER")
    ax.set_xticks(ROLL_OFFS)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.85)
    fig.tight_layout()

    png_path = RESULT_DIR / "rolloff_16qam_ber.png"
    fig.savefig(png_path)
    plt.close(fig)

    print(f"Saved chart: {png_path}")
    print(f"Saved data:  {csv_path}")
    print(f"Saved restored images in: {RESULT_DIR}")


if __name__ == "__main__":
    main()
