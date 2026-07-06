from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .library import BookIndexEntry, BookLibrary


def build_book_summaries(
    session,
    *,
    model: str,
    library: BookLibrary,
    entries: List[BookIndexEntry],
    max_book_chars: int = 12000,
    out_path: Optional[Path] = None,
) -> str:
    """v15.4.0+ default path: summarybook is derived directly from book_overview.json.

    session/model args are kept for compatibility with older callers.
    """
    library.build_summary_files()
    p = out_path or library.summarybook_path()
    return p.read_text(encoding="utf-8") if p.exists() else ""


def build_chapter_summaries_for_book(
    session,
    *,
    model: str,
    library: BookLibrary,
    book: BookIndexEntry,
    max_chapter_chars: int = 16000,
) -> str:
    library.build_summary_files()
    p = library.chapter_summary_path(book.book_id)
    return p.read_text(encoding="utf-8") if p.exists() else ""
