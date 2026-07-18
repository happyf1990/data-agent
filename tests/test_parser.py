from data_agent.embeddings import LocalBGEEmbeddingProvider
from data_agent.models import DocumentMetadata
from data_agent.parser import RuleDocumentParser
from data_agent.qa import build_grounded_prompt
from data_agent.models import QueryResult


def test_chunk_text_uses_overlap():
    parser = RuleDocumentParser(chunk_size=10, chunk_overlap=2)
    chunks = list(parser._chunk_text("abcdefghijklmnopqrstuvwxyz"))
    assert chunks == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"]


def test_section_heading_recognizes_chinese_rule_titles():
    parser = RuleDocumentParser(document_type="policy")
    assert parser._section_heading("第一章 总则") == "第一章 总则"
    assert parser._section_heading("1.2 适用范围") == "1.2 适用范围"


def test_contract_profile_recognizes_contract_clause_titles():
    parser = RuleDocumentParser(document_type="contract")
    assert parser._section_heading("甲方：某某公司") == "甲方：某某公司"
    assert parser._section_heading("违约责任：逾期需赔偿") == "违约责任：逾期需赔偿"


def test_faq_profile_splits_by_questions():
    parser = RuleDocumentParser(document_type="faq")
    blocks = list(parser._split_blocks("Q1: 如何申请？\n提交表单。\nQ2: 多久审批？\n三个工作日。"))
    assert blocks == ["Q1: 如何申请？\n提交表单。", "Q2: 多久审批？\n三个工作日。"]


def test_metadata_payload_contains_trace_fields():
    metadata = DocumentMetadata.from_path("/tmp/rule.pdf", {"department": "hr"})
    from data_agent.models import DocumentChunk

    chunk = DocumentChunk("内容", metadata, 0, section_title="第一章 总则", page_number=3)
    payload = chunk.metadata_payload()
    assert payload["file_name"] == "rule.pdf"
    assert payload["section_title"] == "第一章 总则"
    assert payload["page_number"] == 3
    assert payload["department"] == "hr"


def test_parse_adds_document_type_and_format_metadata():
    parser = RuleDocumentParser(document_type="policy")
    assert parser._resolve_format(__import__("pathlib").Path("rule.pdf"), "auto") == "pdf"


def test_local_bge_extracts_openai_compatible_vectors():
    provider = LocalBGEEmbeddingProvider(api_url="http://localhost:8000/v1/embeddings")
    vectors = provider._extract_vectors({"data": [{"embedding": [0.1, 0.2]}]})
    assert vectors == [[0.1, 0.2]]


def test_build_grounded_prompt_includes_citations():
    prompt = build_grounded_prompt(
        "如何休假？",
        [QueryResult("年假规则", 0.9, {"file_name": "制度.docx", "section_title": "第二条", "page_number": 1})],
    )
    assert "只能依据" in prompt
    assert "[1]" in prompt
    assert "制度.docx" in prompt


def test_manual_profile_recognizes_numbered_manual_titles():
    parser = RuleDocumentParser(document_type="manual")
    assert parser._section_heading("1.1 人员防护") == "1.1 人员防护"
    assert parser._section_heading("9.36.1 货厢拼装工作准备") == "9.36.1 货厢拼装工作准备"


def test_parse_toc_entries_extracts_dotted_leader_toc():
    from data_agent.toc import parse_toc_entries

    entries = parse_toc_entries("1 安全说明 ........ 1-1\n1.1 人员防护 ........ 1-1\n9.36.1 货厢拼装工作准备 ........ 9-107")
    assert [entry.number for entry in entries] == ["1", "1.1", "9.36.1"]
    assert entries[1].title == "人员防护"
    assert entries[2].page_label == "9-107"
    assert entries[2].level == 3


def test_parser_uses_ocr_provider_when_pdf_text_is_empty(monkeypatch):
    from data_agent.ocr import OcrPage

    class FakeOcrProvider:
        def extract_pdf(self, path):
            return [OcrPage(page_number=1, text="1 安全说明\n正文")]

    parser = RuleDocumentParser(document_type="manual", ocr_provider=FakeOcrProvider())
    monkeypatch.setattr(parser, "_read_pdf", lambda path: [(1, "")])
    chunks = parser.parse("manual.pdf", document_format="pdf")
    assert chunks[0].text.startswith("1 安全说明")
    assert chunks[0].metadata.extra["extraction_method"] == "ocr"


def test_local_ocr_extracts_pages_from_api_response():
    from data_agent.ocr import LocalOcrApiProvider

    provider = LocalOcrApiProvider(api_url="http://localhost:8001/ocr/pdf")
    pages = provider._extract_pages({"pages": [{"page_number": 2, "text": "目录", "metadata": {"ocr_engine": "paddleocr"}}]})
    assert pages[0].page_number == 2
    assert pages[0].text == "目录"
    assert pages[0].metadata["ocr_engine"] == "paddleocr"


def test_chunks_to_dify_segments_keeps_content_keywords_and_metadata():
    from data_agent.dify import chunks_to_dify_segments
    from data_agent.models import DocumentChunk

    metadata = DocumentMetadata.from_path("/tmp/manual.pdf", {"document_type": "manual"})
    chunk = DocumentChunk("安全说明", metadata, 0, section_title="1 安全说明", page_number=1)
    segment = chunks_to_dify_segments([chunk])[0]
    assert segment.content == "安全说明"
    assert segment.keywords == ("1 安全说明",)
    assert segment.metadata["document_type"] == "manual"


def test_dify_segment_api_payload_omits_metadata_for_create_segments():
    from data_agent.dify import DifySegment

    payload = DifySegment("内容", keywords=("安全",), metadata={"source": "x"}).to_api_payload()
    assert payload == {"content": "内容", "answer": "", "keywords": ["安全"]}
