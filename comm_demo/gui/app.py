from __future__ import annotations

import sys

from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QApplication

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
