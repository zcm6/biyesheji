from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentOutput:
    """实验运行完成后返回给 GUI 的结果清单。"""

    title: str
    result_dir: Path
    image_paths: list[Path]
    csv_paths: list[Path]
    summary: str = ""
