"""PDF/DOCX rule parser that extracts section-aware chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from data_agent.models import DocumentChunk, DocumentMetadata

DocumentFormat = Literal["auto", "pdf", "docx"]
DocumentType = Literal["generic", "policy", "contract", "faq", "manual"]

SECTION_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百千万0-9]+[章节条款]|[0-9]+(?:\.[0-9]+)*[、.．)]|[一二三四五六七八九十]+[、.．)])\s*(.+)?$"
)
CONTRACT_PATTERN = re.compile(r"^(甲方|乙方|丙方|鉴于|定义|违约责任|争议解决|保密|期限|付款|交付|附件)[：:].*$|^第.+条")
FAQ_PATTERN = re.compile(r"^(Q[0-9]*[:：]|问[:：]|问题[0-9]*[:：])")
MANUAL_PATTERN = re.compile(r"^\d+(?:\.\d+)*\s+.+|^第[一二三四五六七八九十百千万0-9]+[章节条款]")


@dataclass(frozen=True)
class ChunkingProfile:
    """Document-type-specific chunking behavior."""

    name: DocumentType
    chunk_size: int
    chunk_overlap: int
    heading_patterns: tuple[re.Pattern[str], ...]
    split_pattern: re.Pattern[str]


PROFILES: dict[DocumentType, ChunkingProfile] = {
    "generic": ChunkingProfile(
        name="generic",
        chunk_size=900,
        chunk_overlap=120,
        heading_patterns=(SECTION_PATTERN,),
        split_pattern=re.compile(r"\n{2,}|(?<=。)\s*\n"),
    ),
    "policy": ChunkingProfile(
        name="policy",
        chunk_size=800,
        chunk_overlap=120,
        heading_patterns=(SECTION_PATTERN,),
        split_pattern=re.compile(r"\n{2,}|(?<=。)\s*\n|(?=第[一二三四五六七八九十百千万0-9]+[章节条款])"),
    ),
    "contract": ChunkingProfile(
        name="contract",
        chunk_size=700,
        chunk_overlap=100,
        heading_patterns=(SECTION_PATTERN, CONTRACT_PATTERN),
        split_pattern=re.compile(r"\n{2,}|(?=第[一二三四五六七八九十百千万0-9]+条)|(?=甲方[:：])|(?=乙方[:：])"),
    ),
    "faq": ChunkingProfile(
        name="faq",
        chunk_size=600,
        chunk_overlap=80,
        heading_patterns=(FAQ_PATTERN, SECTION_PATTERN),
        split_pattern=re.compile(r"\n{2,}|(?=Q[0-9]*[:：])|(?=问[:：])|(?=问题[0-9]*[:：])"),
    ),
    "manual": ChunkingProfile(
        name="manual",
        chunk_size=900,
        chunk_overlap=120,
        heading_patterns=(MANUAL_PATTERN, SECTION_PATTERN),
        split_pattern=re.compile(r"\n{2,}|(?=\d+(?:\.\d+)*\s+)|(?=第[一二三四五六七八九十百千万0-9]+[章节条款])"),
    ),
}


class RuleDocumentParser:
    """Parse PDF and DOCX rule documents into retrieval-ready chunks.

    ``document_type`` chooses the chunking strategy for different制度文件类型：
    ``policy`` focuses on chapters/articles, ``contract`` keeps clauses and parties
    together, ``faq`` keeps question/answer pairs together, ``manual`` targets numbered
    product/manual sections and OCR table-of-contents text, and ``generic`` is the fallback strategy.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        document_type: DocumentType = "policy",
    ) -> None:
        if document_type not in PROFILES:
            allowed = ", ".join(PROFILES)
            raise ValueError(f"Unsupported document_type: {document_type}. Expected one of: {allowed}")
        self.profile = PROFILES[document_type]
        self.chunk_size = chunk_size if chunk_size is not None else self.profile.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else self.profile.chunk_overlap
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    def parse(
        self,
        path: str | Path,
        extra_metadata: dict | None = None,
        document_format: DocumentFormat = "auto",
    ) -> list[DocumentChunk]:
        """Parse a PDF or DOCX file into chunks with rule metadata."""
        document_path = Path(path)
        resolved_format = self._resolve_format(document_path, document_format)
        metadata_extra = {"document_type": self.profile.name, "document_format": resolved_format}
        metadata_extra.update(extra_metadata or {})
        extraction_method = "text"
        if resolved_format == "pdf":
            pages = self._read_pdf(document_path)
        elif resolved_format == "docx":
            pages = [(None, "\n".join(self._read_docx(document_path)))]
        else:
            raise ValueError(f"Unsupported document format: {resolved_format}")
        metadata_extra["extraction_method"] = extraction_method
        metadata = DocumentMetadata.from_path(document_path, metadata_extra)

        chunks: list[DocumentChunk] = []
        current_section: str | None = None
        for page_number, text in pages:
            for block in self._split_blocks(text):
                heading = self._section_heading(block)
                if heading:
                    current_section = heading
                for piece in self._chunk_text(block):
                    chunks.append(
                        DocumentChunk(
                            text=piece,
                            metadata=metadata,
                            chunk_index=len(chunks),
                            section_title=current_section,
                            page_number=page_number,
                        )
                    )
        return chunks

    def _resolve_format(self, path: Path, document_format: DocumentFormat) -> Literal["pdf", "docx"]:
        if document_format != "auto":
            return document_format
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        if suffix == ".docx":
            return "docx"
        raise ValueError(f"Unsupported document type: {suffix}. Expected .pdf or .docx")

    def _needs_ocr(self, pages: list[tuple[int, str]]) -> bool:
        text_length = sum(len(text.strip()) for _, text in pages)
        return text_length == 0

    def _read_pdf(self, path: Path) -> list[tuple[int, str]]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(index + 1, page.extract_text() or "") for index, page in enumerate(reader.pages)]

    def _read_docx(self, path: Path) -> list[str]:
        from docx import Document

        document = Document(str(path))
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        table_rows = []
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    table_rows.append(" | ".join(cells))
        return paragraphs + table_rows

    def _split_blocks(self, text: str) -> Iterable[str]:
        normalized = re.sub(r"[ \t]+", " ", text).replace("\r\n", "\n")
        for block in self.profile.split_pattern.split(normalized):
            cleaned = block.strip()
            if cleaned:
                yield cleaned

    def _section_heading(self, text: str) -> str | None:
        first_line = text.splitlines()[0].strip()
        for pattern in self.profile.heading_patterns:
            if pattern.match(first_line):
                return first_line
        return None

    def _chunk_text(self, text: str) -> Iterable[str]:
        if len(text) <= self.chunk_size:
            yield text
            return
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            yield text[start:end].strip()
            if end == len(text):
                break
            start = end - self.chunk_overlap
