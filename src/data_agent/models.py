"""Shared data models for rule documents and retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata that is stored together with every chunk."""

    source_path: str
    file_name: str
    file_type: str
    parser_version: str = "rule-parser-v1"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path, extra: dict[str, Any] | None = None) -> "DocumentMetadata":
        document_path = Path(path)
        return cls(
            source_path=str(document_path),
            file_name=document_path.name,
            file_type=document_path.suffix.lower().lstrip("."),
            extra=extra or {},
        )


@dataclass(frozen=True)
class DocumentChunk:
    """A parsed and embeddable rule-document chunk."""

    text: str
    metadata: DocumentMetadata
    chunk_index: int
    section_title: str | None = None
    page_number: int | None = None
    chunk_id: str = field(default_factory=lambda: str(uuid4()))

    def metadata_payload(self) -> dict[str, Any]:
        """Return metadata in a shape that can be serialized into Milvus JSON fields."""
        return {
            "chunk_id": self.chunk_id,
            "source_path": self.metadata.source_path,
            "file_name": self.metadata.file_name,
            "file_type": self.metadata.file_type,
            "parser_version": self.metadata.parser_version,
            "chunk_index": self.chunk_index,
            "section_title": self.section_title,
            "page_number": self.page_number,
            **self.metadata.extra,
        }


@dataclass(frozen=True)
class QueryResult:
    """A knowledge-base search hit used by Q&A applications."""

    text: str
    score: float
    metadata: dict[str, Any]
