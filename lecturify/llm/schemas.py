"""Pydantic schemas for every Gemma JSON response."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SectionOutline(BaseModel):
    title: str
    start_s: float
    end_s: float
    summary: str = ""
    key_concepts: list[str] = Field(default_factory=list)


class CourseOutline(BaseModel):
    course_title: str
    summary: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    sections: list[SectionOutline]


class Formula(BaseModel):
    latex: str
    meaning: str = ""
    source: Literal["image", "transcript"] = "transcript"
    image_index: int | None = None  # 1-based when source == "image"

    @field_validator("source", mode="before")
    @classmethod
    def _norm_source(cls, v):
        v = str(v or "").lower()
        return "image" if "image" in v else "transcript"

    @field_validator("image_index", mode="before")
    @classmethod
    def _norm_index(cls, v):
        try:
            return int(v) if v not in (None, "", "null") else None
        except (TypeError, ValueError):
            return None


class SectionNotes(BaseModel):
    best_image: int = 1
    image_choice_reason: str = ""
    figure_caption: str = ""
    formulas: list[Formula] = Field(default_factory=list)
    notes_markdown: str
    key_points: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- smoke test
class SmokeFormula(BaseModel):
    latex: str
    description: str = ""


class SmokeImages(BaseModel):
    count: int
    descriptions: list[str]


# ---------------------------------------------------------------- screenshots mode
class ImageGroup(BaseModel):
    title: str
    image_indices: list[int]  # 1-based
    summary: str = ""
    key_concepts: list[str] = Field(default_factory=list)


class ImagesOutline(BaseModel):
    course_title: str
    summary: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    sections: list[ImageGroup]
