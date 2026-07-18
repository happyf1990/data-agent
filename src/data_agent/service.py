"""Programmatic interface for ingesting and querying a rule knowledge base."""

from __future__ import annotations

from pathlib import Path

from data_agent.dify import DifyKnowledgeClient, chunks_to_dify_segments, write_dify_jsonl
from data_agent.embeddings import EmbeddingProvider
from data_agent.ocr import OcrProvider
from data_agent.parser import DocumentFormat, DocumentType, RuleDocumentParser
from data_agent.qa import build_grounded_prompt
from data_agent.store import MilvusRuleKnowledgeBase


class RuleKnowledgeService:
    """Small application-facing API around parser + Milvus store."""

    def __init__(self, knowledge_base: MilvusRuleKnowledgeBase) -> None:
        self.knowledge_base = knowledge_base

    @classmethod
    def connect(
        cls,
        milvus_uri: str,
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        milvus_token: str | None = None,
    ) -> "RuleKnowledgeService":
        """Create a service from concrete infrastructure settings."""
        return cls(
            MilvusRuleKnowledgeBase(
                uri=milvus_uri,
                token=milvus_token,
                collection_name=collection_name,
                embedding_provider=embedding_provider,
            )
        )

    def ingest_file(
        self,
        path: str | Path,
        document_type: DocumentType = "policy",
        document_format: DocumentFormat = "auto",
        metadata: dict | None = None,
        ocr_provider: OcrProvider | None = None,
        ocr_on_empty: bool = True,
    ) -> int:
        """Parse one PDF/DOCX file and upsert chunks into the knowledge base."""
        parser = RuleDocumentParser(
            document_type=document_type,
            ocr_provider=ocr_provider,
            ocr_on_empty=ocr_on_empty,
        )
        chunks = parser.parse(path, extra_metadata=metadata, document_format=document_format)
        return self.knowledge_base.upsert_chunks(chunks)

    def export_file_for_dify(
        self,
        path: str | Path,
        output_path: str | Path,
        document_type: DocumentType = "policy",
        document_format: DocumentFormat = "auto",
        metadata: dict | None = None,
        ocr_provider: OcrProvider | None = None,
    ) -> int:
        """Parse one file and write Dify-ready JSONL segments without calling Milvus."""
        parser = RuleDocumentParser(document_type=document_type, ocr_provider=ocr_provider)
        chunks = parser.parse(path, extra_metadata=metadata, document_format=document_format)
        return write_dify_jsonl(chunks, output_path)

    def push_file_to_dify(
        self,
        path: str | Path,
        dify_client: DifyKnowledgeClient,
        dataset_id: str,
        document_name: str | None = None,
        document_type: DocumentType = "policy",
        document_format: DocumentFormat = "auto",
        metadata: dict | None = None,
        ocr_provider: OcrProvider | None = None,
    ) -> dict:
        """Parse one file and create a text document in Dify for RAG indexing."""
        parser = RuleDocumentParser(document_type=document_type, ocr_provider=ocr_provider)
        chunks = parser.parse(path, extra_metadata=metadata, document_format=document_format)
        text = "\n\n".join(chunk.text for chunk in chunks)
        return dify_client.create_document_by_text(
            dataset_id=dataset_id,
            name=document_name or Path(path).name,
            text=text,
            process_rule={"mode": "custom", "rules": {"pre_processing_rules": [], "segmentation": {"separator": "\n\n", "max_tokens": 1000}}},
        )

    def answer_prompt(self, question: str, limit: int = 5, metadata_filter: str | None = None) -> str:
        """Retrieve evidence and return a grounded prompt for an LLM call."""
        contexts = self.knowledge_base.search(question, limit=limit, metadata_filter=metadata_filter)
        return build_grounded_prompt(question, contexts)
