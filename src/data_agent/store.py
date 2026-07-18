"""Milvus storage and retrieval for rule-document chunks."""

from __future__ import annotations

from typing import Iterable

from data_agent.embeddings import EmbeddingProvider
from data_agent.models import DocumentChunk, QueryResult


class MilvusRuleKnowledgeBase:
    """Store parsed rules in Milvus with vector and metadata fields."""

    def __init__(
        self,
        uri: str,
        token: str | None,
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        metric_type: str = "COSINE",
    ) -> None:
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.metric_type = metric_type
        from pymilvus import MilvusClient

        self.client = MilvusClient(uri=uri, token=token)

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        if self.client.has_collection(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedding_provider.dimension,
            metric_type=self.metric_type,
            auto_id=False,
            primary_field_name="chunk_id",
            vector_field_name="embedding",
        )

    def upsert_chunks(self, chunks: Iterable[DocumentChunk]) -> int:
        """Embed and upsert chunks into Milvus, including metadata payloads."""
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        self.ensure_collection()
        embeddings = self.embedding_provider.embed([chunk.text for chunk in chunk_list])
        rows = []
        for chunk, embedding in zip(chunk_list, embeddings, strict=True):
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "embedding": embedding,
                    "text": chunk.text,
                    "metadata": chunk.metadata_payload(),
                }
            )
        self.client.upsert(collection_name=self.collection_name, data=rows)
        return len(rows)

    def search(self, question: str, limit: int = 5, metadata_filter: str | None = None) -> list[QueryResult]:
        """Search rule chunks for a question and return text plus metadata."""
        self.ensure_collection()
        query_vector = self.embedding_provider.embed([question])[0]
        hits = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            filter=metadata_filter,
            output_fields=["text", "metadata"],
        )
        return [
            QueryResult(
                text=hit["entity"].get("text", ""),
                score=float(hit.get("distance", 0.0)),
                metadata=hit["entity"].get("metadata", {}),
            )
            for hit in hits[0]
        ]
