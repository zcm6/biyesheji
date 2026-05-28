from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import DEFAULT_TEXT
from .models import SourceArtifact


def pack_image_data(image: Image.Image) -> tuple[bytes, np.ndarray]:
    """将图像转换为灰度像素载荷。

    Args:
        image: 待处理的 PIL 图像对象。

    Returns:
        一个二元组，包含灰度像素的字节流和 uint8 类型的二维像素数组。
    """
    gray = image.convert("L")
    pixels = np.array(gray, dtype=np.uint8)
    return pixels.tobytes(), pixels


def unpack_image_data(data: bytes, metadata: dict[str, int] | None = None) -> np.ndarray:
    """根据图像元数据将字节载荷还原为灰度图像数组。

    Args:
        data: 图像像素字节载荷；当 ``metadata`` 为空时，前 8 字节会被
            解释为宽度和高度。
        metadata: 图像元数据，包含 ``width`` 和 ``height``。若提供该参数，
            ``data`` 会被视为纯像素载荷。

    Returns:
        uint8 类型的二维灰度图像数组，形状为 ``(height, width)``。

    Raises:
        ValueError: 当图像字节过短或元数据中的宽高无效时抛出。
    """
    if metadata is None:
        if len(data) < 8:
            raise ValueError("image data too short")
        width = int.from_bytes(data[:4], "big")
        height = int.from_bytes(data[4:8], "big")
        payload = data[8:]
    else:
        width = int(metadata.get("width", 0))
        height = int(metadata.get("height", 0))
        payload = data
    if width <= 0 or height <= 0:
        raise ValueError("invalid image metadata")
    total = width * height
    pixels = np.frombuffer(payload, dtype=np.uint8).copy()
    if len(pixels) < total:
        pixels = np.concatenate([pixels, np.zeros(total - len(pixels), dtype=np.uint8)])
    elif len(pixels) > total:
        pixels = pixels[:total]
    return pixels.reshape(height, width)


def pack_audio_data(samples: np.ndarray, sample_rate: int) -> bytes:
    """将 PCM 音频采样压缩为 8 位量化载荷。

    Args:
        samples: 原始音频采样数组，通常为 int16 PCM 数据。
        sample_rate: 音频采样率。该参数用于保持接口语义一致，实际采样率
            通过带外元数据传递。

    Returns:
        int8 量化后的音频字节载荷。
    """
    mono = samples.astype(np.int16)
    quantized = np.clip(np.round(mono / 256.0), -128, 127).astype(np.int8)
    return quantized.tobytes()


def unpack_audio_data(data: bytes, metadata: dict[str, int] | None = None) -> tuple[int, np.ndarray]:
    """根据音频元数据将字节载荷还原为 PCM 采样。

    Args:
        data: 音频量化字节载荷；当 ``metadata`` 为空时，前 8 字节会被
            解释为采样率和采样点数。
        metadata: 音频元数据，包含 ``sample_rate`` 和 ``count``。若提供
            该参数，``data`` 会被视为纯音频载荷。

    Returns:
        一个二元组，包含采样率和 int16 类型的一维 PCM 采样数组。

    Raises:
        ValueError: 当音频字节过短或元数据中的采样率、采样点数无效时抛出。
    """
    if metadata is None:
        if len(data) < 8:
            raise ValueError("audio data too short")
        sample_rate = int.from_bytes(data[:4], "big")
        count = int.from_bytes(data[4:8], "big")
        payload = data[8:]
    else:
        sample_rate = int(metadata.get("sample_rate", 0))
        count = int(metadata.get("count", 0))
        payload = data
    if sample_rate <= 0 or count < 0:
        raise ValueError("invalid audio metadata")
    quantized = np.frombuffer(payload, dtype=np.int8).copy()
    if len(quantized) < count:
        quantized = np.concatenate([quantized, np.zeros(count - len(quantized), dtype=np.int8)])
    elif len(quantized) > count:
        quantized = quantized[:count]
    samples = (quantized.astype(np.int16) * 256).astype(np.int16)
    return sample_rate, samples


def pcm_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """将 PCM 采样封装为 WAV 格式字节数据。

    Args:
        samples: 待封装的一维 PCM 采样数组。
        sample_rate: 音频采样率。

    Returns:
        单声道、16-bit 采样宽度的 WAV 文件字节数据。
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


def prepare_source(kind: str, text: str, path: str) -> SourceArtifact:
    """根据输入类型准备通信链路的信源数据。

    Args:
        kind: 信源类型，支持 ``"文本"``、``"图像"`` 和语音文件类型。
        text: 文本信源内容；为空时使用默认文本。
        path: 图像或语音文件路径。

    Returns:
        SourceArtifact 对象，包含统一的原始字节载荷、预览信息以及媒体元数据。

    Raises:
        ValueError: 当文件路径无效或语音文件不是 16-bit PCM WAV 时抛出。
    """
    if kind == "文本":
        content = (text.strip() or DEFAULT_TEXT).replace("\r\n", "\n").replace("\r", "\n")
        return SourceArtifact(kind=kind, raw_bytes=content.encode("utf-8"), preview_text=content, text_value=content)
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError("请先选择有效文件。")
    if kind == "图像":
        image = Image.open(file_path)
        raw_bytes, pixels = pack_image_data(image) # raw_bytes = 图片二进制流 pixels = 图片像素矩阵
        return SourceArtifact(
            kind=kind,
            raw_bytes=raw_bytes,
            preview_text=f"{file_path.name}\n尺寸: {pixels.shape[1]} x {pixels.shape[0]}",
            image_array=pixels,
            media_meta={"width": int(pixels.shape[1]), "height": int(pixels.shape[0])},
        )
    with wave.open(str(file_path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError("当前仅支持 16-bit PCM WAV 音频。")
        frames = wav_file.readframes(wav_file.getnframes()) # 将所有帧转化为音频文件的原始字节数据
        samples = np.frombuffer(frames, dtype=np.int16) # 将字节数据解析为 int16 类型的 PCM 采样数组
        if wav_file.getnchannels() > 1:
            samples = samples.reshape(-1, wav_file.getnchannels())[:, 0]
        payload = pack_audio_data(samples, wav_file.getframerate()) # 将 PCM 采样压缩为 8 位量化载荷，得到音频的字节载荷
        return SourceArtifact(
            kind=kind, 
            raw_bytes=payload,
            preview_text=f"{file_path.name}\n采样率: {wav_file.getframerate()} Hz\n时长: {len(samples) / wav_file.getframerate():.2f} s",
            audio_samples=samples.astype(np.float32),
            sample_rate=wav_file.getframerate(),
            audio_wav_bytes=pcm_to_wav_bytes(samples, wav_file.getframerate()),
            media_meta={"sample_rate": int(wav_file.getframerate()), "count": int(len(samples))},
        )


def restore_output(
    kind: str, restored_bytes: bytes, metadata: dict[str, int] | None = None
) -> tuple[str, np.ndarray | None, np.ndarray | None, int, bytes | None]:
    """将恢复出的字节数据转换为可展示的信源结果。

    Args:
        kind: 原始信源类型，用于选择文本、图像或语音的恢复方式。
        restored_bytes: 信源解码后得到的字节数据。
        metadata: 图像或语音恢复所需的带外媒体元数据。

    Returns:
        一个五元组，依次为预览文本、恢复图像数组、恢复音频采样数组、
        音频采样率和 WAV 格式音频字节。与当前 ``kind`` 无关的结果项
        会返回 ``None`` 或 0。
    """
    if kind == "文本":
        return restored_bytes.decode("utf-8", errors="replace"), None, None, 0, None
    if kind == "图像":
        try:
            image = unpack_image_data(restored_bytes, metadata)
            return f"图像恢复成功\n尺寸: {image.shape[1]} x {image.shape[0]}", image, None, 0, None
        except Exception:
            return "图像恢复失败，可能由信道误码导致。", None, None, 0, None
    try:
        sample_rate, samples = unpack_audio_data(restored_bytes, metadata)
        wav_bytes = pcm_to_wav_bytes(samples, sample_rate)
        return f"语音恢复完成\n采样率: {sample_rate} Hz\n采样点数: {len(samples)}", None, samples.astype(np.float32), sample_rate, wav_bytes
    except Exception:
        return "语音恢复失败，输出数据为空或已损坏。", None, None, 0, None
