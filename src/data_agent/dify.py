"""Local Dify RAG export helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from data_agent.models import DocumentChunk


@dataclass(frozen=True)
class DifySegment:
    """A Dify-friendly local segment exported from a parsed chunk."""

    content: str
    keywords: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    def to_jsonl_payload(self) -> dict[str, Any]:
        """Return a metadata-rich JSONL row for local upload/import workflows."""
        return {
            "content": self.content,
            "keywords": list(self.keywords),
            "metadata": self.metadata or {},
        }


def chunks_to_dify_segments(chunks: Iterable[DocumentChunk], keyword_fields: tuple[str, ...] = ("section_title",)) -> list[DifySegment]:
    """Convert parsed chunks into local Dify-oriented segment rows."""
    segments: list[DifySegment] = []
    for chunk in chunks:
        metadata = chunk.metadata_payload()
        keywords = tuple(str(metadata[field]) for field in keyword_fields if metadata.get(field))
        segments.append(DifySegment(content=chunk.text, keywords=keywords, metadata=metadata))
    return segments


def write_dify_jsonl(chunks: Iterable[DocumentChunk], output_path: str | Path) -> int:
    """Write chunks as JSONL for local review before uploading to Dify."""
    segments = chunks_to_dify_segments(chunks)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for segment in segments:
            file.write(json.dumps(segment.to_jsonl_payload(), ensure_ascii=False) + "\n")
    return len(segments)


def write_dify_text(chunks: Iterable[DocumentChunk], output_path: str | Path) -> int:
    """Write chunks as one local text file that can be uploaded to Dify."""
    chunk_list = list(chunks)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunk_list:
            metadata = chunk.metadata_payload()
            section = metadata.get("section_title") or "未识别章节"
            page = metadata.get("page_number") or "N/A"
            file.write(f"# {section} | page={page}\n")
            file.write(chunk.text.strip() + "\n\n")
    return len(chunk_list)
