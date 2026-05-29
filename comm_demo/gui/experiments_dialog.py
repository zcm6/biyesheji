from __future__ import annotations

import contextlib
import io
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class ExperimentSpec:
    title: str
    module_name: str
    description: str


EXPERIMENTS = (
    ExperimentSpec(
        title="调制方式 BER 对比",
        module_name="experiments.experiment_modulation_semilogy",
        description="比较 16ASK、16PSK、16QAM 在 AWGN 信道下的 BER 曲线。",
    ),
    ExperimentSpec(
        title="信道编码 BER 对比",
        module_name="experiments.experiment_channel_code_semilogy",
        description="比较 CRC、汉明码、卷积码在 16QAM/AWGN 链路下的 BER 曲线。",
    ),
    ExperimentSpec(
        title="格雷码映射 BER 对比",
        module_name="experiments.experiment_gray_ber",
        description="比较自然映射和格雷码映射对 BER 的影响。",
    ),
    ExperimentSpec(
        title="AI-BiGRU 与 Viterbi BER 对比",
        module_name="experiments.experiment_ai_viterbi_media_ber",
        description="比较 AI-BiGRU 与硬判决 Viterbi 在媒体传输任务下的 BER，耗时较长。",
    ),
)


class ExperimentWorker(QObject):
    finished = pyqtSignal(object, str)
    failed = pyqtSignal(str, str)
    progress = pyqtSignal(str)

    def __init__(self, spec: ExperimentSpec):
        super().__init__()
        self.spec = spec

    def run(self) -> None:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                module = import_module(self.spec.module_name)
                output = module.run_experiment(progress_callback=self.progress.emit)
        except Exception as exc:
            self.failed.emit(str(exc), buffer.getvalue())
            return
        self.finished.emit(output, buffer.getvalue())


class ImagePreview(QLabel):
    def __init__(self):
        super().__init__("实验完成后显示结果图")
        self._source_pixmap = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._source_pixmap = QPixmap()
            self.setPixmap(QPixmap())
            self.setText(f"结果图加载失败:\n{path}")
            return
        self._source_pixmap = pixmap
        self.setText("")
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


class ExperimentsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("实验中心")
        self.resize(1080, 760)
        self.worker_thread: QThread | None = None
        self.worker: ExperimentWorker | None = None
        self.current_output = None
        self.experiment_buttons: list[QPushButton] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QGroupBox("实验项目")
        left_layout = QVBoxLayout(left)
        for spec in EXPERIMENTS:
            button = QPushButton(spec.title)
            button.clicked.connect(lambda _checked=False, item=spec: self._start_experiment(item))
            self.experiment_buttons.append(button)
            left_layout.addWidget(button)
            hint = QLabel(spec.description)
            hint.setWordWrap(True)
            left_layout.addWidget(hint)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.title_label = QLabel("请选择一个实验")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size:16px; padding:6px;")
        self.image_tabs = QTabWidget()
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        self.log_text.setPlainText("点击左侧实验按钮后，将在后台运行实验并显示生成的图像与数据文件。")
        right_layout.addWidget(self.title_label)
        right_layout.addWidget(self.image_tabs, 1)
        right_layout.addWidget(self.log_text)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([300, 780])

        bottom = QHBoxLayout()
        self.status_label = QLabel("就绪")
        self.export_button = QPushButton("导出结果")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_current_result)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        bottom.addWidget(self.status_label, 1)
        bottom.addWidget(self.export_button)
        bottom.addWidget(close_button)
        root.addLayout(bottom)

    def _start_experiment(self, spec: ExperimentSpec) -> None:
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.information(self, "提示", "当前实验仍在运行，请稍候。")
            return
        self.current_output = None
        self.export_button.setEnabled(False)
        self.image_tabs.clear()
        self.title_label.setText(spec.title)
        self.log_text.setPlainText("实验运行中，请稍候...")
        self.status_label.setText(f"正在运行: {spec.title}")
        self._set_busy(True)

        self.worker_thread = QThread(self)
        self.worker = ExperimentWorker(spec)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._append_log)
        self.worker.finished.connect(self._handle_experiment_finished)
        self.worker.failed.connect(self._handle_experiment_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self._handle_thread_finished)
        self.worker_thread.start()

    def _append_log(self, message: str) -> None:
        if message:
            self.log_text.appendPlainText(message)

    def _handle_experiment_finished(self, output, logs: str) -> None:
        self.current_output = output
        self._show_output(output, logs)
        self.status_label.setText(f"完成: {output.title}")

    def _handle_experiment_failed(self, error: str, logs: str) -> None:
        text = logs.strip()
        if text:
            text += "\n\n"
        text += f"实验运行失败:\n{error}"
        self.log_text.setPlainText(text)
        self.status_label.setText("实验失败")
        QMessageBox.critical(self, "实验失败", error)

    def _handle_thread_finished(self) -> None:
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
        self.worker_thread = None
        self.worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        for button in self.experiment_buttons:
            button.setEnabled(not busy)
        self.export_button.setEnabled(self.current_output is not None and not busy)

    def _show_output(self, output, logs: str) -> None:
        self.image_tabs.clear()
        for image_path in output.image_paths:
            preview = ImagePreview()
            preview.set_image(Path(image_path))
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(preview)
            self.image_tabs.addTab(scroll, self._tab_title(Path(image_path)))
        if self.image_tabs.count() == 0:
            empty = QLabel("该实验没有生成可预览图片")
            empty.setAlignment(Qt.AlignCenter)
            self.image_tabs.addTab(empty, "结果")

        lines = [
            output.summary,
            "",
            f"输出目录: {output.result_dir}",
            "",
            "图像文件:",
            *[f"- {path}" for path in output.image_paths],
            "",
            "数据文件:",
            *[f"- {path}" for path in output.csv_paths],
        ]
        if logs.strip():
            lines.extend(["", "运行日志:", logs.strip()])
        self.log_text.setPlainText("\n".join(lines))

    def _tab_title(self, path: Path) -> str:
        name = path.stem.lower()
        if "16ask" in name:
            return "16ASK"
        if "16psk" in name:
            return "16PSK"
        if "16qam" in name:
            return "16QAM"
        return "结果图"

    def _export_current_result(self) -> None:
        if self.current_output is None:
            QMessageBox.information(self, "提示", "请先运行一个实验。")
            return
        directory = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not directory:
            return
        target_dir = Path(directory)
        copied: list[str] = []
        for source in [*self.current_output.image_paths, *self.current_output.csv_paths]:
            source_path = Path(source)
            if not source_path.exists():
                continue
            target_path = target_dir / source_path.name
            if source_path.resolve() != target_path.resolve():
                shutil.copy2(source_path, target_path)
            copied.append(target_path.name)
        summary_path = target_dir / f"{self.current_output.title}_实验说明.txt"
        summary_path.write_text(self.log_text.toPlainText(), encoding="utf-8")
        copied.append(summary_path.name)
        self.status_label.setText(f"已导出 {len(copied)} 个文件到: {target_dir}")

    def closeEvent(self, event):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.information(self, "提示", "实验仍在运行，完成后再关闭窗口。")
            event.ignore()
            return
        super().closeEvent(event)
