from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from PyQt5.QtCore import QObject, Qt, QThread, QTimer, QUrl, pyqtSignal 
from PyQt5.QtGui import QFont, QFontDatabase, QImage, QPixmap
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from comm_demo.pipeline import DEFAULT_TEXT, SPS, SimulationResult, SimulationSession, create_session

# 将NumPy数组（灰度图像数据）转换为PyQt的QPixmap，以便在图形界面显示图像
def array_to_pixmap(array: np.ndarray) -> QPixmap:
    image = np.ascontiguousarray(array.astype(np.uint8))
    height, width = image.shape
    qimage = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimage.copy())

# 将WAV格式的字节数据写入临时文件，并返回文件路径，以便QMediaPlayer播放音频
def write_temp_wav(wav_bytes: bytes | None, prefix: str) -> str | None:
    if not wav_bytes:
        return None
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", delete=False)
    handle.write(wav_bytes)
    handle.flush()
    handle.close()
    return handle.name

# 定义一个名为 PlotCanvas 的类，用于在 GUI环境中嵌入 Matplotlib 绘图功能
class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        super().__init__(self.figure)

"""定义一个名为 StepWorker 的类，继承自 QObject，用于在单独的线程中执行仿真步骤，并通过信号与主线程通信"""
class StepWorker(QObject):
    finished = pyqtSignal(str, object) # 发射完成信号，包含步骤说明和结果
    failed = pyqtSignal(str) # 发射失败信号，包含错误信息

    def __init__(self, session: SimulationSession):
        super().__init__()
        self.session = session

    def run(self):
        try:
            note = self.session.step()
            result = self.session.build_result() if self.session.is_finished() else None
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(note, result)

# 配置matplotlib的字体设置
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
        signal = np.real(matched_signal[: 400 * SPS])
        for index in range(pulse_length - 1, max(0, len(signal) - span), SPS):
            axis.plot(signal[index : index + span], color="#0a6", alpha=0.15)
        axis.set_title("眼图")
        axis.grid(alpha=0.3)
        canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于 Python 的通信系统仿真与教学演示系统")
        self.resize(1560, 980) # 更主窗口大小
        self.session: SimulationSession | None = None
        self.result: SimulationResult | None = None
        self.eye_dialog: EyeDialog | None = None
        self.input_audio_path: str | None = None
        self.output_audio_path: str | None = None
        self.worker_thread: QThread | None = None
        self.worker: StepWorker | None = None
        self.player = QMediaPlayer(self)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._advance_one_step)
        self._build_ui()
        self._sync_mode()
        self._sync_order()
        self._sync_channel_params()
        self._draw_media_waveforms()
        self._draw_idle_views()
        self._refresh_action_buttons()
        QTimer.singleShot(0, self._init_right_splitter_sizes)
        self.statusBar().showMessage("就绪")

    def _build_ui(self):
        # 构建主界面，使用水平分割器将左侧的控制面板和右侧的可视化区分开
        wrapper = QWidget()
        self.setCentralWidget(wrapper)
        root = QHBoxLayout(wrapper)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # 左侧控制面板包含信源输入、参数配置、流程控制、核心指标和阶段回溯等功能模块
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_source_box())
        left_layout.addWidget(self._build_config_box())
        left_layout.addWidget(self._build_control_box())
        left_layout.addWidget(self._build_metrics_box())
        left_layout.addWidget(self._build_history_box())

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.addWidget(self._build_media_box())
        self.right_splitter.addWidget(self._build_view_box())
        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 1)
        self.right_splitter.setChildrenCollapsible(False)
        self.right_splitter.setSizes([1, 1])  # 初始化上下区域 50% : 50%
        right_layout.addWidget(self.right_splitter)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setSizes([450, 1110])

    def _init_right_splitter_sizes(self):
        if not hasattr(self, "right_splitter"):
            return
        total = self.right_splitter.size().height()
        if total <= 0:
            self.right_splitter.setSizes([1, 1])
            return
        half = total // 2
        self.right_splitter.setSizes([half, total - half])

    # 构建主界面左侧的”输入与预处理“参数面板
    def _build_source_box(self) -> QWidget:
        box = QGroupBox("输入与预处理")
        form = QFormLayout(box)
        self.kind = QComboBox()
        self.kind.addItems(["文本", "图像", "语音"])
        self.kind.currentTextChanged.connect(self._sync_mode)
        self.text_input = QPlainTextEdit(DEFAULT_TEXT)
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        browse = QPushButton("选择文件")
        browse.clicked.connect(self._browse_file)
        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(browse)
        self.source_hint = QLabel("请选择信源类型并输入文本或导入文件。")
        self.source_hint.setWordWrap(True)
        self.input_preview = QTextEdit()
        self.input_preview.setReadOnly(True)
        self.input_preview.setMinimumHeight(120)
        form.addRow("输入类型", self.kind)
        form.addRow("文本输入", self.text_input)
        form.addRow("文件路径", file_row)
        form.addRow("操作提示", self.source_hint)
        form.addRow("输入预览", self.input_preview)
        return box

    # 构建主界面左侧的”参数配置“参数面板
    def _build_config_box(self) -> QWidget:
        box = QGroupBox("参数配置")
        form = QFormLayout(box)
        self.source_method = QComboBox()
        self.source_method.addItems(["哈夫曼编码", "香农-范诺编码", "算术编码"])
        self.channel_method = QComboBox()
        self.channel_method.addItems(["CRC", "汉明码", "卷积码"])
        self.modulation = QComboBox()
        self.modulation.addItems(["MASK", "MPSK", "MQAM"])
        self.modulation.currentTextChanged.connect(self._sync_order) # 根据调制方式同步可选的调制阶数
        self.order = QComboBox()
        self.channel_name = QComboBox()
        self.channel_name.addItems(["AWGN", "瑞利衰落", "莱斯衰落"])
        self.channel_name.currentTextChanged.connect(self._sync_channel_params) # 根据信道模型同步相关参数的启用状态和提示信息
        self.gray_option = QComboBox()
        self.gray_option.addItems(["\u5426", "\u662f"])
        self.snr = QLineEdit("12")
        self.kfactor = QLineEdit("3")
        self.roll_off = QLineEdit("0.35")
        # 将上述参数配置项以标签-控件的形式添加到表单布局中，形成一个整齐的参数配置界面
        for label, widget in [
            ("信源编码", self.source_method),
            ("信道编码", self.channel_method),
            ("调制方式", self.modulation),
            ("调制阶数", self.order),
            ("信道模型", self.channel_name),
            ("\u683c\u96f7\u7801\u7f16\u7801", self.gray_option),
            ("SNR(dB)", self.snr),
            ("莱斯 K", self.kfactor),
            ("滚降系数", self.roll_off),
        ]:
            form.addRow(label, widget)
        return box

    # 构建主界面左侧的”流程控制“操作面板
    def _build_control_box(self) -> QWidget:
        box = QGroupBox("流程控制")
        grid = QGridLayout(box)
        self.start_button = QPushButton("开始/继续")
        self.start_button.clicked.connect(self._start_or_resume) 
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self._pause)
        self.step_button = QPushButton("单步执行")
        self.step_button.clicked.connect(self._step_once)
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self._reset)
        self.eye_button = QPushButton("查看眼图")
        self.eye_button.clicked.connect(self._open_eye_dialog)
        self.export_button = QPushButton("导出结果")
        self.export_button.clicked.connect(self._export)
        for index, button in enumerate(
            [self.start_button, self.pause_button, self.step_button, self.reset_button, self.eye_button, self.export_button]
        ):
            grid.addWidget(button, index // 2, index % 2)
        return box

    # 构建主界面左侧的”核心指标“显示面板
    def _build_metrics_box(self) -> QWidget:
        box = QGroupBox("核心指标")
        layout = QVBoxLayout(box)
        self.ber_label = QLabel("BER: -")
        self.ber_label.setAlignment(Qt.AlignCenter)
        self.ber_label.setStyleSheet("font-size:16px; padding:10px; border:1px solid #ccc;")
        self.metrics = QTextEdit()
        self.metrics.setReadOnly(True)
        self.metrics.setMinimumHeight(150)
        layout.addWidget(self.ber_label)
        layout.addWidget(self.metrics)
        return box

    # 构建主界面左侧的”阶段回溯“显示面板
    def _build_history_box(self) -> QWidget:
        box = QGroupBox("阶段回溯")
        layout = QVBoxLayout(box)
        self.stage_history = QTextEdit()
        self.stage_history.setReadOnly(True)
        layout.addWidget(self.stage_history)
        return box

    # 构建主界面右侧的”原始输入与恢复输出“显示面板，包括文本、图像和语音的预览，以及相关的播放按钮
    def _build_media_box(self) -> QWidget:
        box = QGroupBox("原始输入与恢复输出")
        layout = QVBoxLayout(box)

        media_row = QWidget()
        media_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        media_layout = QHBoxLayout(media_row)
        media_layout.setContentsMargins(0, 0, 0, 0)

        original_box = QGroupBox("原始媒体")
        original_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        original_layout = QVBoxLayout(original_box)
        self.original_text = QTextEdit()
        self.original_text.setReadOnly(True)
        self.original_image = QLabel("原始图像预览")
        self.original_image.setAlignment(Qt.AlignCenter)
        self.original_image.setMinimumHeight(160)
        self.original_image.setStyleSheet("border:1px solid #ccc;background:#f8f8f8;")
        self.play_input_audio_button = QPushButton("播放原始语音")
        self.play_input_audio_button.clicked.connect(lambda: self._play_audio(self.input_audio_path))
        original_layout.addWidget(self.original_text)
        original_layout.addWidget(self.original_image)
        original_layout.addWidget(self.play_input_audio_button)

        restored_box = QGroupBox("恢复媒体")
        restored_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        restored_layout = QVBoxLayout(restored_box)
        self.restored_text = QTextEdit()
        self.restored_text.setReadOnly(True)
        self.restored_image = QLabel("恢复图像预览")
        self.restored_image.setAlignment(Qt.AlignCenter)
        self.restored_image.setMinimumHeight(160)
        self.restored_image.setStyleSheet("border:1px solid #ccc;background:#f8f8f8;")
        self.play_output_audio_button = QPushButton("播放恢复语音")
        self.play_output_audio_button.clicked.connect(lambda: self._play_audio(self.output_audio_path))
        restored_layout.addWidget(self.restored_text)
        restored_layout.addWidget(self.restored_image)
        restored_layout.addWidget(self.play_output_audio_button)

        media_layout.addWidget(original_box, 1)
        media_layout.addWidget(restored_box, 1)

        self.media_wave_canvas = PlotCanvas()
        layout.addWidget(media_row, 1)
        layout.addWidget(self.media_wave_canvas, 1)
        return box

    # 构建主界面右侧的”可视化区“显示面板，包括比特序列、时域/频域信号、星座图和眼图等多个标签页，以及相关的选择器用于切换不同阶段的显示内容
    def _build_view_box(self) -> QWidget:
        box = QGroupBox("可视化区")
        layout = QVBoxLayout(box)
        selectors = QWidget()
        selectors_layout = QHBoxLayout(selectors)
        selectors_layout.setContentsMargins(0, 0, 0, 0)
        self.bit_selector = QComboBox()
        self.bit_selector.addItems(["原始比特流", "信源编码后", "信道编码后", "判决后比特流", "信道解码后"])
        self.bit_selector.currentTextChanged.connect(self._redraw_views)
        self.signal_selector = QComboBox()
        self.signal_selector.addItems(["调制后信号", "信道后信号", "匹配滤波后信号"])
        self.signal_selector.currentTextChanged.connect(self._redraw_views)
        selectors_layout.addWidget(QLabel("比特阶段"))
        selectors_layout.addWidget(self.bit_selector)
        selectors_layout.addWidget(QLabel("信号阶段"))
        selectors_layout.addWidget(self.signal_selector)

        self.tabs = QTabWidget()
        self.bit_canvas = PlotCanvas()
        self.signal_canvas = PlotCanvas()
        self.const_canvas = PlotCanvas()
        self.eye_canvas = PlotCanvas()
        for title, canvas in [
            ("比特序列", self.bit_canvas),
            ("时域/频域", self.signal_canvas),
            ("星座图", self.const_canvas),
            ("眼图", self.eye_canvas),
        ]:
            self.tabs.addTab(canvas, title)
        layout.addWidget(selectors)
        layout.addWidget(self.tabs, 1)
        return box

    # 根据当前选择的输入类型（文本、图像或语音）同步界面元素的状态和提示信息
    def _sync_mode(self):
        kind = self.kind.currentText()
        self.text_input.setEnabled(kind == "文本")
        self.file_path.setEnabled(kind != "文本")
        if kind == "文本":
            self.source_hint.setText("请输入中英文文本，系统会将其转换为比特流。")
            self.input_preview.setPlainText(self.text_input.toPlainText().strip() or DEFAULT_TEXT)
        elif kind == "图像":
            self.source_hint.setText("请选择图像文件，系统会先灰度化再转换为比特流。")
            self.input_preview.setPlainText(self.file_path.text() or "请选择图像文件。")
        else:
            self.source_hint.setText("请选择 16-bit PCM WAV 音频文件，系统会读取采样并量化。")
            self.input_preview.setPlainText(self.file_path.text() or "请选择语音文件。")

    def _sync_order(self):
        self.order.clear()
        if self.modulation.currentText() == "MQAM":
            self.order.addItems(["16"])
        else:
            self.order.addItems(["2", "4", "8", "16"])

    def _sync_channel_params(self):
        is_rician = self.channel_name.currentText() == "莱斯衰落"
        self.kfactor.setEnabled(is_rician)
        self.kfactor.setPlaceholderText("" if is_rician else "仅莱斯衰落有效")
        self.kfactor.setToolTip("K 值仅在莱斯衰落信道下生效。")

    def _browse_file(self):
        kind = self.kind.currentText()
        file_filter = "Images (*.png *.jpg *.jpeg *.bmp)" if kind == "图像" else "Wave (*.wav)"
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if not path:
            return
        self.file_path.setText(path)
        self.input_preview.setPlainText(path)
        if kind == "图像":
            image = Image.open(path).convert("L")
            pixmap = array_to_pixmap(np.array(image, dtype=np.uint8))
            self.original_image.setPixmap(
                pixmap.scaled(self.original_image.width(), self.original_image.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        self.statusBar().showMessage(f"已选择文件: {Path(path).name}")

    def _collect_session(self) -> SimulationSession:
        channel_name = self.channel_name.currentText()
        gray_ok = self.gray_option.currentIndex() == 1
        return create_session(
            kind=self.kind.currentText(),
            text=self.text_input.toPlainText(),
            path=self.file_path.text(),
            source_method=self.source_method.currentText(),
            channel_method=self.channel_method.currentText(),
            modulation=self.modulation.currentText(),
            order=int(self.order.currentText()),
            channel_name=channel_name,
            snr_db=float(self.snr.text()),
            k_factor=float(self.kfactor.text()) if channel_name == "莱斯衰落" else 3.0,
            roll_off=float(self.roll_off.text()),
            gray_ok=gray_ok,
        )

    def _ensure_session(self):
        if self.session is None or self.session.is_finished():
            self.session = self._collect_session()
            self.result = None
            self.stage_history.clear()
            self.metrics.clear()

    def _start_or_resume(self):
        # 如果仿真正在运行或有步骤正在执行，则不执行任何操作
        if self.timer.isActive() or self.worker_thread is not None:
            return
        if self.session is None or self.session.is_finished():
            self._cleanup_temp_audio() # 清理之前的临时音频文件
            self._reset_runtime_views() # 重置可视化区的显示状态
            try:
                self.session = self._collect_session()
            except Exception as exc:
                QMessageBox.critical(self, "参数错误", str(exc))
                return
        self.timer.start()
        self._refresh_action_buttons()
        self.statusBar().showMessage("仿真运行中")

    def _pause(self):
        self.timer.stop()
        self._refresh_action_buttons()
        if self.worker_thread is not None:
            self.statusBar().showMessage("当前步骤执行中，本步完成后暂停")
        else:
            self.statusBar().showMessage("已暂停")

    def _step_once(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.worker_thread is not None:
            return
        self._ensure_session()
        self._advance_one_step()

    def _advance_one_step(self):
        if self.session is None or self.worker_thread is not None:
            return
        self.statusBar().showMessage(f"执行中: {self.session.next_stage()}")
        self._run_step_async()

    def _run_step_async(self):
        if self.session is None or self.worker_thread is not None:
            return
        self.worker_thread = QThread(self)
        self.worker = StepWorker(self.session)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_step_finished)
        self.worker.failed.connect(self._handle_step_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._on_worker_thread_finished)
        self.worker_thread.start()
        self._refresh_action_buttons()

    def _handle_step_finished(self, note: str, result: SimulationResult | None):
        self.stage_history.setPlainText("\n".join(self.session.history))
        self.statusBar().showMessage(note)
        self._render_source_preview()
        self._redraw_partial_views()
        if result is not None:
            self.timer.stop()
            self.result = result
            self._render_result()
        else:
            self._render_partial_metrics()

    def _handle_step_failed(self, error: str):
        self.timer.stop()
        self._refresh_action_buttons()
        QMessageBox.critical(self, "执行失败", error)

    def _on_worker_thread_finished(self):
        self.worker_thread = None
        self.worker = None
        self._refresh_action_buttons()

    def _refresh_action_buttons(self):
        busy = self.worker_thread is not None
        self.start_button.setEnabled(not busy and not self.timer.isActive())
        self.step_button.setEnabled(not busy)
        self.reset_button.setEnabled(not busy)
        self.pause_button.setEnabled(self.timer.isActive() or busy)
        self.export_button.setEnabled(self.result is not None and not busy)
        self.eye_button.setEnabled(self.result is not None and not busy)

    def _render_partial_metrics(self):
        if self.session is None:
            return
        lines = [
            f"当前阶段: {self.session.next_stage()}",
            f"已完成阶段数: {self.session.stage_index}/8",
        ]
        if self.session.source_bits is not None:
            lines.append(f"原始比特数: {len(self.session.source_bits)}")
        if self.session.source_coded_bits is not None:
            lines.append(f"信源编码后比特数: {len(self.session.source_coded_bits)}")
        if self.session.channel_coded_bits is not None:
            lines.append(f"信道编码后比特数: {len(self.session.channel_coded_bits)}")
        self.metrics.setPlainText("\n".join(lines))

    def _render_source_preview(self):
        if self.session is None or self.session.source is None:
            return
        source = self.session.source
        self.original_text.setPlainText(source.preview_text)
        if source.kind == "文本":
            self.original_image.setPixmap(QPixmap())
            self.original_image.setText("文本模式下无图像预览")
            self.play_input_audio_button.setEnabled(False)
            self.input_audio_path = None
        elif source.kind == "图像":
            self.original_image.setText("")
            self.original_image.setPixmap(
                array_to_pixmap(source.image_array).scaled(
                    self.original_image.width(),
                    self.original_image.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            self.play_input_audio_button.setEnabled(False)
            self.input_audio_path = None
        else:
            self.original_image.setPixmap(QPixmap())
            self.original_image.setText("语音模式下请查看下方波形并点击播放")
            if self.input_audio_path and Path(self.input_audio_path).exists():
                try:
                    os.remove(self.input_audio_path)
                except OSError:
                    pass
            self.input_audio_path = write_temp_wav(source.audio_wav_bytes, "comm_demo_input_")
            self.play_input_audio_button.setEnabled(self.input_audio_path is not None)
        self._draw_media_waveforms()

    def _render_result(self):
        if self.result is None:
            return
        summary = self.result.summary_text()
        self.metrics.setPlainText(summary)
        self.stage_history.setPlainText("\n".join(self.result.history))
        self.restored_text.setPlainText(self.result.restored_text)
        ber_color = "#1f8b4c" if self.result.ber < 0.05 else "#b94a48"
        self.ber_label.setText(f"BER: {self.result.ber:.6f}")
        self.ber_label.setStyleSheet(f"font-size:16px; padding:10px; border:1px solid #ccc; color:{ber_color};")
        if self.result.restored_image is not None:
            self.restored_image.setText("")
            self.restored_image.setPixmap(
                array_to_pixmap(self.result.restored_image).scaled(
                    self.restored_image.width(),
                    self.restored_image.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            self.restored_image.setPixmap(QPixmap())
            self.restored_image.setText("当前结果无恢复图像")
        if self.output_audio_path and Path(self.output_audio_path).exists():
            try:
                os.remove(self.output_audio_path)
            except OSError:
                pass
        self.output_audio_path = write_temp_wav(self.result.restored_audio_wav_bytes, "comm_demo_output_")
        self.play_output_audio_button.setEnabled(self.output_audio_path is not None)
        self._draw_media_waveforms()
        self._redraw_views()
        self.statusBar().showMessage("仿真完成")

    def _draw_media_waveforms(self):
        fig = self.media_wave_canvas.figure
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1)
        if self.session and self.session.source and self.session.source.audio_samples is not None:
            samples = self.session.source.audio_samples
            rate = self.session.source.sample_rate
            x = np.arange(len(samples)) / max(rate, 1)
            ax1.plot(x[: min(len(x), 4000)], samples[: min(len(samples), 4000)], color="#1f77b4")
            ax1.set_title("原始语音波形")
            ax1.set_xlabel("时间(s)")
            ax1.grid(alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "当前输入不是语音", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_axis_off()
        if self.result and self.result.restored_audio_samples is not None:
            samples = self.result.restored_audio_samples
            rate = self.result.restored_audio_rate
            x = np.arange(len(samples)) / max(rate, 1)
            ax2.plot(x[: min(len(x), 4000)], samples[: min(len(samples), 4000)], color="#d62728")
            ax2.set_title("恢复语音波形")
            ax2.set_xlabel("时间(s)")
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "当前结果没有恢复语音", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_axis_off()
        self.media_wave_canvas.draw()

    def _draw_idle_views(self):
        bit_fig = self.bit_canvas.figure
        bit_fig.clear()
        ax = bit_fig.subplots(1, 1)
        ax.text(0.5, 0.5, "运行仿真后显示比特序列", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("比特序列")
        ax.set_axis_off()
        self.bit_canvas.draw()

        signal_fig = self.signal_canvas.figure
        signal_fig.clear()
        ax1, ax2 = signal_fig.subplots(2, 1)
        ax1.text(0.5, 0.5, "运行到成型调制后显示时域波形", ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "运行到成型调制后显示频域频谱", ha="center", va="center", transform=ax2.transAxes)
        ax1.set_axis_off()
        ax2.set_axis_off()
        self.signal_canvas.draw()

        const_fig = self.const_canvas.figure
        const_fig.clear()
        ax = const_fig.subplots(1, 1)
        ax.text(0.5, 0.5, "运行到成型调制后显示星座图", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("星座图")
        ax.set_axis_off()
        self.const_canvas.draw()

        eye_fig = self.eye_canvas.figure
        eye_fig.clear()
        ax = eye_fig.subplots(1, 1)
        ax.text(0.5, 0.5, "运行到匹配滤波与判决后显示眼图", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("眼图")
        ax.set_axis_off()
        self.eye_canvas.draw()

    def _redraw_views(self):
        if self.result is None:
            return
        self._draw_views_from_data(
            self.result.bit_views(),
            self.result.signal_views(),
            self.result.tx_symbols,
            self.result.sampled_symbols,
            self.result.matched_signal,
        )

    def _redraw_partial_views(self):
        if self.session is None:
            return
        bit_views = {}
        signal_views = {}
        if self.session.source_bits is not None:
            bit_views["原始比特流"] = self.session.source_bits
        if self.session.source_coded_bits is not None:
            bit_views["信源编码后"] = self.session.source_coded_bits
        if self.session.channel_coded_bits is not None:
            bit_views["信道编码后"] = self.session.channel_coded_bits
        if self.session.rx_channel_bits is not None:
            bit_views["判决后比特流"] = self.session.rx_channel_bits
        if self.session.decoded_source_bits is not None:
            bit_views["信道解码后"] = self.session.decoded_source_bits
        if self.session.tx_signal is not None:
            signal_views["调制后信号"] = self.session.tx_signal
        if self.session.rx_signal is not None:
            signal_views["信道后信号"] = self.session.rx_signal
        if self.session.matched_signal is not None:
            signal_views["匹配滤波后信号"] = self.session.matched_signal
        self._draw_views_from_data(
            bit_views,
            signal_views,
            self.session.tx_symbols,
            self.session.sampled_symbols,
            self.session.matched_signal,
        )

    """"根据提供的比特视图、信号视图、发送符号、采样符号和匹配滤波后信号数据，更新可视化区的比特序列、时域/频域波形、星座图和眼图等显示内容"""
    def _draw_views_from_data(
        self,
        bit_views: dict[str, np.ndarray],
        signal_views: dict[str, np.ndarray],
        tx_symbols: np.ndarray | None,
        sampled_symbols: np.ndarray | None,
        matched_signal: np.ndarray | None,
    ):
        # 画比特序列图
        bit_fig = self.bit_canvas.figure
        bit_fig.clear()
        ax = bit_fig.subplots(1, 1)
        selected_bits = bit_views.get(self.bit_selector.currentText())
        if selected_bits is not None:
            limit = min(256, len(selected_bits))
            ax.step(np.arange(limit), selected_bits[:limit], where="post")
            ax.set_ylim(-0.2, 1.2)
            ax.set_title(self.bit_selector.currentText())
            # ax.axis("equal")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "当前阶段暂无该比特数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        self.bit_canvas.draw()

        # 画出信号的时域波形和频域频谱
        signal_fig = self.signal_canvas.figure
        signal_fig.clear()
        ax1, ax2 = signal_fig.subplots(2, 1)
        selected_signal = signal_views.get(self.signal_selector.currentText())
        if selected_signal is not None and len(selected_signal) > 0:
            count = min(600, len(selected_signal))
            ax1.plot(np.real(selected_signal[:count]), label="实部") # 画出实部波形
            if np.max(np.abs(np.imag(selected_signal[:count]))) > 1e-9:
                ax1.plot(np.imag(selected_signal[:count]), label="虚部", alpha=0.8) # 如果虚部不全为零，则画出虚部波形
            ax1.set_title(f"{self.signal_selector.currentText()}时域波形")
            ax1.grid(alpha=0.3)
            ax1.legend()
            spectrum = np.fft.fftshift(np.fft.fft(selected_signal[: min(len(selected_signal), 4096)], n=2048))
            freq = np.linspace(-0.5, 0.5, len(spectrum))
            ax2.plot(freq, 20 * np.log10(np.abs(spectrum) + 1e-9))
            ax2.set_title(f"{self.signal_selector.currentText()}频谱")
            # ax.axis("equal")
            ax2.grid(alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "当前阶段暂无该信号数据", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_axis_off()
            ax2.set_axis_off()
        self.signal_canvas.draw()

        # 画出星座图，发送符号和接收采样分别用不同的颜色表示
        const_fig = self.const_canvas.figure
        const_fig.clear()
        ax = const_fig.subplots(1, 1)
        if tx_symbols is not None and len(tx_symbols) > 0:
            tx = tx_symbols[:500]
            ax.scatter(np.real(tx), np.imag(tx), s=12, alpha=0.6, label="发送符号")
            if sampled_symbols is not None and len(sampled_symbols) > 0:
                rx = sampled_symbols[:500]
                ax.scatter(np.real(rx), np.imag(rx), s=12, alpha=0.6, label="接收采样")
            ax.set_title("星座图")
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            ax.axis("equal")
            ax.grid(alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "当前阶段暂无星座数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        self.const_canvas.draw()

        eye_fig = self.eye_canvas.figure
        eye_fig.clear()
        ax = eye_fig.subplots(1, 1)
        if matched_signal is not None and len(matched_signal) > 0:
            # signal = np.real(matched_signal)
            signal = np.real(matched_signal[: 400 * SPS])
            span = 2 * SPS
            for index in range(len(self.session.pulse) - 1, max(0, len(signal) - span), SPS):
                ax.plot(signal[index : index + span], color="#0a6", alpha=0.15)
            ax.set_title("眼图")
            # ax.axis("equal")
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "当前阶段暂无眼图数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        self.eye_canvas.draw()

    def _play_audio(self, path: str | None):
        if not path or not Path(path).exists():
            QMessageBox.information(self, "提示", "当前没有可播放的语音文件。")
            return
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(str(Path(path).resolve()))))
        self.player.play()
        self.statusBar().showMessage(f"正在播放: {Path(path).name}")

    def _open_eye_dialog(self):
        if self.result is None:
            QMessageBox.information(self, "提示", "请先运行一次完整仿真。")
            return
        self.eye_dialog = EyeDialog(self.result.matched_signal, len(self.result.pulse))
        self.eye_dialog.show()

    def _export(self):
        if self.result is None:
            QMessageBox.information(self, "提示", "请先运行一次完整仿真。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出结果", "simulation_report.txt", "Text (*.txt)")
        if not path:
            return
        content = (
            self.result.summary_text()
            + "\n\n阶段历史:\n"
            + "\n".join(self.result.history)
            + "\n\n原始预览:\n"
            + self.original_text.toPlainText()
            + "\n\n恢复预览:\n"
            + self.restored_text.toPlainText()
        )
        Path(path).write_text(content, encoding="utf-8")
        self.statusBar().showMessage(f"已导出到: {Path(path).name}")

    def _reset_runtime_views(self):
        self.stage_history.clear()
        self.metrics.clear()
        self.original_text.clear()
        self.restored_text.clear()
        self.original_image.setPixmap(QPixmap())
        self.original_image.setText("原始图像预览")
        self.restored_image.setPixmap(QPixmap())
        self.restored_image.setText("恢复图像预览")
        self.ber_label.setText("BER: -")
        self.ber_label.setStyleSheet("font-size:16px; padding:10px; border:1px solid #ccc;")
        self._draw_media_waveforms()
        self._draw_idle_views()

    def _cleanup_temp_audio(self):
        for attr in ["input_audio_path", "output_audio_path"]:
            path = getattr(self, attr)
            if path and Path(path).exists():
                try:
                    os.remove(path)
                except OSError:
                    pass
            setattr(self, attr, None)

    def _reset(self):
        if self.worker_thread is not None:
            QMessageBox.information(self, "提示", "当前步骤仍在执行，请等待完成后再重置。")
            return
        self.timer.stop()
        self.player.stop()
        self.session = None
        self.result = None
        self._cleanup_temp_audio()
        self.file_path.clear()
        self.input_preview.clear()
        self._reset_runtime_views()
        self.play_input_audio_button.setEnabled(False)
        self.play_output_audio_button.setEnabled(False)
        self._refresh_action_buttons()
        self.statusBar().showMessage("已重置")
        self._sync_mode()

    def closeEvent(self, event):
        self.timer.stop()
        self.player.stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self._cleanup_temp_audio()
        super().closeEvent(event)


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
