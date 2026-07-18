import json
from pathlib import Path


def test_pdf_to_milvus_and_dify_notebook_is_valid_and_reviewable():
    notebook_path = Path("notebooks/pdf_to_milvus_and_dify_rag.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "PaddleOCR" in sources
    assert "LocalBGEEmbeddingProvider" in sources
    assert "MilvusRuleKnowledgeBase" in sources
    assert "write_dify_jsonl" in sources
    assert "write_dify_text" in sources
    assert "PDF_FILES" in sources
