"""Command-line interface for parsing and searching rule documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_agent.embeddings import HashEmbeddingProvider, LocalBGEEmbeddingProvider
from data_agent.parser import RuleDocumentParser
from data_agent.qa import build_grounded_prompt
from data_agent.store import MilvusRuleKnowledgeBase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF/DOCX 规则解析 + Milvus 知识库存储工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="解析 PDF/DOCX 并写入 Milvus")
    ingest.add_argument("paths", nargs="+", type=Path, help="PDF 或 DOCX 文件路径")
    ingest.add_argument("--milvus-uri", default="http://localhost:19530")
    ingest.add_argument("--milvus-token")
    ingest.add_argument("--collection", default="rule_knowledge_base")
    ingest.add_argument("--embedding", choices=["bge", "hash"], default="bge")
    ingest.add_argument("--bge-api-url", help="本地 BGE embedding API 地址，默认读取 BGE_API_URL 或 http://localhost:8000/v1/embeddings")
    ingest.add_argument("--bge-model", default="bge-m3")
    ingest.add_argument("--bge-dimension", type=int, default=1024)
    ingest.add_argument("--document-type", choices=["generic", "policy", "contract", "faq"], default="policy", help="制度/规则文件类型，用于选择对应切割策略")
    ingest.add_argument("--document-format", choices=["auto", "pdf", "docx"], default="auto", help="显式指定解析格式；auto 根据扩展名判断")
    ingest.add_argument("--metadata", default="{}", help="追加元数据 JSON，例如 '{\"dept\":\"hr\"}'")

    search = subparsers.add_parser("search", help="检索知识库并输出问答提示词")
    search.add_argument("question")
    search.add_argument("--milvus-uri", default="http://localhost:19530")
    search.add_argument("--milvus-token")
    search.add_argument("--collection", default="rule_knowledge_base")
    search.add_argument("--embedding", choices=["bge", "hash"], default="bge")
    search.add_argument("--bge-api-url", help="本地 BGE embedding API 地址，默认读取 BGE_API_URL 或 http://localhost:8000/v1/embeddings")
    search.add_argument("--bge-model", default="bge-m3")
    search.add_argument("--bge-dimension", type=int, default=1024)
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--filter", dest="metadata_filter")
    return parser


def _embedding_provider(args):
    if args.embedding == "bge":
        return LocalBGEEmbeddingProvider(
            api_url=args.bge_api_url,
            model=args.bge_model,
            dimension=args.bge_dimension,
        )
    return HashEmbeddingProvider()


def main() -> None:
    args = build_parser().parse_args()
    kb = MilvusRuleKnowledgeBase(
        uri=args.milvus_uri,
        token=args.milvus_token,
        collection_name=args.collection,
        embedding_provider=_embedding_provider(args),
    )
    if args.command == "ingest":
        parser = RuleDocumentParser(document_type=args.document_type)
        extra_metadata = json.loads(args.metadata)
        total = 0
        for path in args.paths:
            chunks = parser.parse(path, extra_metadata=extra_metadata, document_format=args.document_format)
            total += kb.upsert_chunks(chunks)
        print(json.dumps({"upserted": total}, ensure_ascii=False))
        return
    results = kb.search(args.question, limit=args.limit, metadata_filter=args.metadata_filter)
    print(build_grounded_prompt(args.question, results))


if __name__ == "__main__":
    main()
