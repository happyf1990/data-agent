"""Dify RAG export and Knowledge API helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib import request

from data_agent.models import DocumentChunk


@dataclass(frozen=True)
class DifySegment:
    """A Dify segment payload derived from a local document chunk."""

    content: str
    answer: str = ""
    keywords: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    def to_api_payload(self) -> dict[str, Any]:
        """Return the shape expected by Dify's Create Chunks API."""
        payload: dict[str, Any] = {"content": self.content, "answer": self.answer}
        if self.keywords:
            payload["keywords"] = list(self.keywords)
        return payload

    def to_jsonl_payload(self) -> dict[str, Any]:
        """Return a metadata-rich JSONL row for offline Dify import workflows."""
        payload = self.to_api_payload()
        payload["metadata"] = self.metadata or {}
        return payload


class DifyKnowledgeClient:
    """Minimal Dify Knowledge API client for text documents and custom chunks."""

    def __init__(self, api_base_url: str, api_key: str, timeout_seconds: float = 60.0) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def create_document_by_text(
        self,
        dataset_id: str,
        name: str,
        text: str,
        doc_form: str = "text_model",
        doc_language: str = "Chinese",
        indexing_technique: str | None = "high_quality",
        process_rule: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a Dify document from text and return Dify's response."""
        payload: dict[str, Any] = {
            "name": name,
            "text": text,
            "doc_form": doc_form,
            "doc_language": doc_language,
        }
        if indexing_technique:
            payload["indexing_technique"] = indexing_technique
        if process_rule:
            payload["process_rule"] = process_rule
        return self._post(f"/datasets/{dataset_id}/document/create-by-text", payload)

    def create_segments(self, dataset_id: str, document_id: str, segments: Iterable[DifySegment]) -> dict[str, Any]:
        """Create one or more custom chunks in an existing Dify document."""
        payload = {"segments": [segment.to_api_payload() for segment in segments]}
        return self._post(f"/datasets/{dataset_id}/documents/{document_id}/segments", payload)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            f"{self.api_base_url}{path}",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def chunks_to_dify_segments(chunks: Iterable[DocumentChunk], keyword_fields: tuple[str, ...] = ("section_title",)) -> list[DifySegment]:
    """Convert parsed chunks into Dify custom segment payloads."""
    segments: list[DifySegment] = []
    for chunk in chunks:
        metadata = chunk.metadata_payload()
        keywords = tuple(str(metadata[field]) for field in keyword_fields if metadata.get(field))
        segments.append(DifySegment(content=chunk.text, keywords=keywords, metadata=metadata))
    return segments


def write_dify_jsonl(chunks: Iterable[DocumentChunk], output_path: str | Path) -> int:
    """Write chunks as metadata-rich JSONL for Dify-oriented RAG ingestion."""
    segments = chunks_to_dify_segments(chunks)
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as file:
        for segment in segments:
            file.write(json.dumps(segment.to_jsonl_payload(), ensure_ascii=False) + "\n")
    return len(segments)
