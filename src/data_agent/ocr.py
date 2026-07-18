"""OCR providers for scanned/image-like PDF documents."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib import request


@dataclass(frozen=True)
class OcrPage:
    """OCR text and optional layout metadata for one PDF page."""

    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class OcrProvider(Protocol):
    """Protocol implemented by OCR backends."""

    def extract_pdf(self, path: str | Path) -> list[OcrPage]:
        """Extract OCR text for every page in a PDF."""


class LocalOcrApiProvider:
    """OCR provider backed by a local HTTP API.

    The API receives ``{"path": "/absolute/file.pdf"}`` and should return either
    ``{"pages": [{"page_number": 1, "text": "...", "metadata": {...}}]}``
    or ``{"text": "..."}`` for single-document responses.
    """

    def __init__(self, api_url: str | None = None, timeout_seconds: float = 120.0) -> None:
        self.api_url = api_url or os.getenv("OCR_API_URL", "http://localhost:8001/ocr/pdf")
        self.timeout_seconds = timeout_seconds

    def extract_pdf(self, path: str | Path) -> list[OcrPage]:
        document_path = Path(path).expanduser().resolve()
        payload = json.dumps({"path": str(document_path)}, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        return self._extract_pages(body)

    def _extract_pages(self, body: dict[str, Any]) -> list[OcrPage]:
        if "pages" in body:
            return [
                OcrPage(
                    page_number=int(page.get("page_number") or page.get("page") or index + 1),
                    text=page.get("text", ""),
                    metadata=page.get("metadata", {}),
                )
                for index, page in enumerate(body["pages"])
            ]
        if "text" in body:
            return [OcrPage(page_number=1, text=body["text"], metadata=body.get("metadata", {}))]
        raise ValueError("OCR API response must contain either 'pages' or 'text'")
