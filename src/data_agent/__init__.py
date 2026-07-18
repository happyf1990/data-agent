"""Rule document parser and Milvus-backed knowledge base."""

from data_agent.dify import DifyKnowledgeClient, DifySegment, chunks_to_dify_segments, write_dify_jsonl
from data_agent.embeddings import LocalBGEEmbeddingProvider
from data_agent.models import DocumentChunk, DocumentMetadata, QueryResult
from data_agent.service import RuleKnowledgeService
from data_agent.toc import TocEntry, parse_toc_entries
from data_agent.ocr import LocalOcrApiProvider, OcrPage
from data_agent.parser import RuleDocumentParser
from data_agent.store import MilvusRuleKnowledgeBase

__all__ = [
    "DocumentChunk",
    "DifyKnowledgeClient",
    "DifySegment",
    "DocumentMetadata",
    "LocalBGEEmbeddingProvider",
    "LocalOcrApiProvider",
    "MilvusRuleKnowledgeBase",
    "OcrPage",
    "QueryResult",
    "RuleDocumentParser",
    "RuleKnowledgeService",
    "TocEntry",
    "chunks_to_dify_segments",
    "parse_toc_entries",
    "write_dify_jsonl",
]
