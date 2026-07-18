"""Utilities for parsing table-of-contents pages from OCR/PDF text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TOC_LINE_PATTERN = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*)\s+(?P<title>.+?)\s*[.·…]{2,}\s*(?P<page>[IVXLCDM]+|\d+(?:-\d+)?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TocEntry:
    """A normalized table-of-contents entry."""

    number: str
    title: str
    page_label: str
    level: int
    children: tuple["TocEntry", ...] = field(default_factory=tuple)


def parse_toc_entries(text: str) -> list[TocEntry]:
    """Parse dotted-leader TOC text into flat entries with hierarchy levels.

    This is useful after OCR for scanned/image-like PDF目录 pages such as
    ``1.1 人员防护 ........ 1-1``. The returned entries can be saved as metadata
    or used to route downstream section-level parsing.
    """
    entries: list[TocEntry] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        match = TOC_LINE_PATTERN.match(line)
        if not match:
            continue
        number = match.group("number")
        entries.append(
            TocEntry(
                number=number,
                title=match.group("title").strip(),
                page_label=match.group("page"),
                level=number.count(".") + 1,
            )
        )
    return entries
