from __future__ import annotations

import tempfile

import numpy as np
from PyQt5.QtGui import QImage, QPixmap

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
