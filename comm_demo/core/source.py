from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from .constants import DEFAULT_TEXT
from .models import SourceArtifact

def pack_image_data(image: Image.Image) -> tuple[bytes, np.ndarray]:
    gray = image.convert("L")
    pixels = np.array(gray, dtype=np.uint8)
    # 元数据在带外传递；信道仅携带有效载荷。
    return pixels.tobytes(), pixels


def unpack_image_data(data: bytes, metadata: dict[str, int] | None = None) -> np.ndarray:
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
    mono = samples.astype(np.int16)
    quantized = np.clip(np.round(mono / 256.0), -128, 127).astype(np.int8)
    #元数据在带外传递；信道仅携带有效载荷。
    return quantized.tobytes()


def unpack_audio_data(data: bytes, metadata: dict[str, int] | None = None) -> tuple[int, np.ndarray]:
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
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()


# ==================== 信源输入处理 ====================
def prepare_source(kind: str, text: str, path: str) -> SourceArtifact:
    """
    准备信源数据：根据类型（文本/图像/语音）读取并打包数据
    统一转换为字节流，便于后续数字处理
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
        frames = wav_file.readframes(wav_file.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16)
        if wav_file.getnchannels() > 1:
            samples = samples.reshape(-1, wav_file.getnchannels())[:, 0]
        payload = pack_audio_data(samples, wav_file.getframerate())
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
