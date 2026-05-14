from __future__ import annotations

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

ANALYSIS_TIME_WINDOW = 4000
ANALYSIS_TIME_PLOT_POINTS = 600
ANALYSIS_FREQ_WINDOW = 16384
ANALYSIS_WELCH_SEGMENT = 2048
ANALYSIS_WELCH_OVERLAP = 1024
ANALYSIS_NFFT = 8192
AUDIO_WAVEFORM_POINTS = 4000

# 定义一个名为 PlotCanvas 的类，用于在 GUI环境中嵌入 Matplotlib 绘图功能
class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(self.figure)

def configure_plot_fonts(preferred_fonts: list[str]) -> None:
    usable: list[str] = []
    for family in preferred_fonts:
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        usable.append(family)
    if not usable:
        return
    existing = list(rcParams.get("font.sans-serif", []))
    rcParams["font.family"] = ["sans-serif"]
    rcParams["font.sans-serif"] = usable + [family for family in existing if family not in usable]
    rcParams["axes.unicode_minus"] = False


def _center_window(signal: np.ndarray, length: int) -> np.ndarray:
    if signal is None or len(signal) == 0:
        return np.zeros(0, dtype=np.complex64)
    if len(signal) <= length:
        return signal
    start = max(0, (len(signal) - length) // 2)
    return signal[start : start + length]


def _downsample_for_plot(signal: np.ndarray, max_points: int) -> np.ndarray:
    if signal is None or len(signal) <= max_points:
        return signal
    step = int(np.ceil(len(signal) / max_points))
    return signal[::step]


def _random_sample_for_plot(signal: np.ndarray, max_points: int, seed: int = 20240518) -> np.ndarray:
    if signal is None or len(signal) <= max_points:
        return signal
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(signal), size=max_points, replace=False)
    return signal[np.sort(indices)]

# 返回有效语音附近的固定长度窗口及其在原始序列中的起点。
def _audio_activity_window(
    samples: np.ndarray, max_points: int = AUDIO_WAVEFORM_POINTS, start: int | None = None
) -> tuple[np.ndarray, int]:
    if samples is None or len(samples) <= max_points:
        return samples, 0
    data = np.asarray(samples)
    if start is not None:
        start = max(0, min(int(start), len(data) - max_points))
        return data[start : start + max_points], start
    magnitude = np.abs(data)
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    if peak <= 1e-9:
        return data[:max_points], 0
    smooth_len = max(32, min(256, max_points // 20))
    smooth = np.convolve(magnitude, np.ones(smooth_len, dtype=np.float32) / smooth_len, mode="same")
    active = np.flatnonzero(smooth > peak * 0.03)
    if active.size == 0:
        return data[:max_points], 0
    start = max(0, int(active[0]) - max_points // 10)
    end = min(len(data), start + max_points)
    start = max(0, end - max_points)
    return data[start:end], start


def _welch_spectrum_db(
    signal: np.ndarray,
    nfft: int = ANALYSIS_NFFT,
    segment_len: int = ANALYSIS_WELCH_SEGMENT,
    overlap: int = ANALYSIS_WELCH_OVERLAP,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal)
    if x.size == 0:
        return np.linspace(-0.5, 0.5, nfft, endpoint=False), np.full(nfft, -120.0, dtype=np.float32)

    if x.size < segment_len:
        padded = np.zeros(segment_len, dtype=np.complex64)
        padded[: x.size] = x
        x = padded

    hop = max(1, segment_len - overlap)
    window = np.hanning(segment_len).astype(np.float32)
    window_energy = float(np.sum(window * window) + 1e-12)

    pxx = np.zeros(nfft, dtype=np.float64)
    segments = 0
    for start in range(0, x.size - segment_len + 1, hop):
        seg = x[start : start + segment_len]
        spec = np.fft.fftshift(np.fft.fft(seg * window, n=nfft))
        pxx += (np.abs(spec) ** 2) / window_energy
        segments += 1

    if segments == 0:
        spec = np.fft.fftshift(np.fft.fft(x[:segment_len] * window, n=nfft))
        pxx = (np.abs(spec) ** 2) / window_energy
    else:
        pxx /= segments

    freq = np.linspace(-0.5, 0.5, nfft, endpoint=False)
    db = 10 * np.log10(pxx + 1e-12)
    return freq, db.astype(np.float32, copy=False)
