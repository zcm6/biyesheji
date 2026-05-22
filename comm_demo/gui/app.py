from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication


def _configure_runtime_cache() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "comm_demo_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)


_configure_runtime_cache()

from .main_window import MainWindow
from .plotting import configure_plot_fonts

def run_app():
    app = QApplication(sys.argv)
    preferred_fonts = [
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "PingFang SC",
        "Hiragino Sans GB",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "SimSun",
    ]
    configure_plot_fonts(preferred_fonts)
    available = set(QFontDatabase().families())
    for family in preferred_fonts:
        if family in available:
            app.setFont(QFont(family, 10))
            break
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
