from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QMainWindow

from comm_demo.core.constants import SPS
from .plotting import PlotCanvas, _center_window

"""显示独立的眼图绘制窗口"""
class EyeDialog(QMainWindow):
    def __init__(self, matched_signal: np.ndarray, pulse_length: int):
        super().__init__()
        self.setWindowTitle("独立眼图窗口")
        self.resize(920, 520)
        canvas = PlotCanvas()
        self.setCentralWidget(canvas)
        axis = canvas.figure.subplots(1, 1)
        span = 2 * SPS
        steady = _center_window(np.asarray(matched_signal), 400 * SPS)
        signal = np.real(steady)
        for index in range(pulse_length - 1, max(0, len(signal) - span), SPS):
            axis.plot(signal[index : index + span], color="#0a6", alpha=0.15)
        axis.set_title("眼图")
        axis.grid(alpha=0.3)
        canvas.draw()
