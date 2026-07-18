"""Utilities for building intelligent Q&A prompts from retrieved rules."""

from __future__ import annotations

from data_agent.models import QueryResult


def build_grounded_prompt(question: str, contexts: list[QueryResult]) -> str:
    """Build a grounded Q&A prompt that cites retrieved rule chunks."""
    evidence = []
    for index, context in enumerate(contexts, start=1):
        metadata = context.metadata
        source = metadata.get("file_name", "unknown")
        section = metadata.get("section_title") or "未识别章节"
        page = metadata.get("page_number") or "N/A"
        evidence.append(f"[{index}] 来源={source}; 章节={section}; 页码={page}\n{context.text}")
    joined = "\n\n".join(evidence) or "无可用上下文"
    return (
        "你是规则制度问答助手。只能依据下列上下文回答；如果上下文不足，请说明无法确定。\n\n"
        f"问题：{question}\n\n"
        f"上下文：\n{joined}\n\n"
        "请给出简洁答案，并在关键结论后标注引用编号。"
    )
