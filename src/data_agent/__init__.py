"""Notebook-friendly PDF/DOCX parsing, Milvus storage, and Dify local export."""

from data_agent.dify import DifySegment, chunks_to_dify_segments, write_dify_jsonl, write_dify_text
from data_agent.embeddings import HashEmbeddingProvider, LocalBGEEmbeddingProvider
from data_agent.models import DocumentChunk, DocumentMetadata, QueryResult
from data_agent.parser import RuleDocumentParser
from data_agent.qa import build_grounded_prompt
from data_agent.store import MilvusRuleKnowledgeBase
from data_agent.toc import TocEntry, parse_toc_entries

__all__ = [
    "DifySegment",
    "DocumentChunk",
    "DocumentMetadata",
    "HashEmbeddingProvider",
    "LocalBGEEmbeddingProvider",
    "MilvusRuleKnowledgeBase",
    "QueryResult",
    "RuleDocumentParser",
    "TocEntry",
    "build_grounded_prompt",
    "chunks_to_dify_segments",
    "parse_toc_entries",
    "write_dify_jsonl",
    "write_dify_text",
]
