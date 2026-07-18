"""Embedding providers for the rule knowledge base."""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Protocol
from urllib import request


class EmbeddingProvider(Protocol):
    """Protocol implemented by embedding backends."""

    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class LocalBGEEmbeddingProvider:
    """Embedding provider backed by a local BGE HTTP API.

    The provider expects an OpenAI-compatible embeddings endpoint by default,
    for example a local bge-m3 service at ``http://localhost:8000/v1/embeddings``.
    """

    def __init__(
        self,
        api_url: str | None = None,
        model: str = "bge-m3",
        dimension: int = 1024,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_url = api_url or os.getenv("BGE_API_URL", "http://localhost:8000/v1/embeddings")
        self.model = model
        self.dimension = dimension
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        vectors = self._extract_vectors(body)
        if len(vectors) != len(texts):
            raise ValueError(f"BGE API returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors

    def _extract_vectors(self, body: dict[str, Any]) -> list[list[float]]:
        if "data" in body:
            return [item["embedding"] for item in body["data"]]
        if "embeddings" in body:
            return body["embeddings"]
        raise ValueError("BGE API response must contain either 'data' or 'embeddings'")


class HashEmbeddingProvider:
    """Deterministic local embedding provider for tests only."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
