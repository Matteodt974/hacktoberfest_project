"""Central configuration for Lecturify."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SUPPORTED_MODELS = ("gemma-4-26b-a4b-it", "gemma-4-31b-it")
DEFAULT_MODEL = SUPPORTED_MODELS[0]

LANGUAGES = {"en": "English", "fr": "French"}


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    lang: str = "en"
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("LECTURIFY_DATA", ROOT / "data")))

    # Video limits (seconds)
    warn_duration_s: int = 20 * 60
    max_duration_s: int = 45 * 60

    # Frame sampling
    sample_every_s: float = 1.0
    thumb_width: int = 160
    motion_threshold: float = 2.0
    min_edge_density: float = 0.01
    group_phash_dist: int = 6
    global_phash_dist: int = 4
    subsume_window_s: float = 20.0
    subsume_containment: float = 0.85
    max_frames: int = 150
    min_frames: int = 5
    fallback_every_s: float = 10.0
    frame_width: int = 1024
    jpeg_quality: int = 85

    # LLM
    max_images_per_section: int = 6
    concurrency: int = 3
    max_retries: int = 7
    thinking_outline: str = "high"
    thinking_notes: str = "high"

    force: bool = False
    verify: bool = False  # Pass 3: Chain-of-Verification
    ablation: bool = False  # same notes WITHOUT images, to measure what seeing the frames adds

    @property
    def language_name(self) -> str:
        return LANGUAGES.get(self.lang, "English")

    def video_dir(self, video_id: str) -> Path:
        d = self.data_dir / video_id
        d.mkdir(parents=True, exist_ok=True)
        return d
