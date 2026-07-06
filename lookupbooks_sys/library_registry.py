from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .library import BookLibrary, BookIndexEntry

REGISTRY_FILENAME = "library_registry.txt"


@dataclass
class RegistryRow:
    book_id: int
    title: str
    source_dir: str
    has_overview: bool
    has_sections: bool
    chapter_count: int
    section_count: int
    chapter_summary_ok: bool
    quality_label: str
    recommendation: str
    library_ready: bool
    notes: str


def registry_path(library_root: Path) -> Path:
    return Path(library_root) / REGISTRY_FILENAME


def compute_registry_rows(library_root: Path) -> Tuple[List[RegistryRow], Dict[str, str]]:
    lib = BookLibrary(Path(library_root))
    entries = lib.load_index()
    rows: List[RegistryRow] = []
    sb_ok = lib.summarybook_path().exists() and lib.summarybook_path().stat().st_size > 10
    for e in entries:
        cs = lib.chapter_summary_path(e.book_id)
        chapter_summary_ok = cs.exists() and cs.stat().st_size > 10
        notes: List[str] = []
        if not e.has_overview:
            notes.append("missing_overview")
        if not e.has_sections:
            notes.append("missing_sections")
        if not chapter_summary_ok:
            notes.append("missing_chapter_summary")
        if getattr(e, "embedded_in_library", False):
            notes.append("embedded_copy")
        library_ready = bool(e.has_overview and e.has_sections)
        rows.append(
            RegistryRow(
                book_id=e.book_id,
                title=e.title,
                source_dir=e.source_dir,
                has_overview=e.has_overview,
                has_sections=e.has_sections,
                chapter_count=e.chapter_count,
                section_count=e.section_count,
                chapter_summary_ok=chapter_summary_ok,
                quality_label=e.quality_label,
                recommendation=e.recommendation,
                library_ready=library_ready,
                notes=";".join(notes),
            )
        )
    meta = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "summarybook_ok": "1" if sb_ok else "0",
        "books_indexed": str(len(entries)),
    }
    return rows, meta


def write_registry(library_root: Path, rows: List[RegistryRow], meta: Dict[str, str]) -> Path:
    p = registry_path(Path(library_root))
    p.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# lookupbooks_sys library registry",
        f"# generated_at: {meta.get('generated_at','')}",
        f"# summarybook_ok: {meta.get('summarybook_ok','0')}",
        f"# books_indexed: {meta.get('books_indexed','0')}",
        "# columns: book_id\ttitle\tsource_dir\thas_overview\thas_sections\tchapter_count\tsection_count\tchapter_summary_ok\tquality_label\trecommendation\tlibrary_ready\tnotes",
    ]
    lines: List[str] = []
    for r in sorted(rows, key=lambda x: x.book_id):
        lines.append(
            "\t".join(
                [
                    str(r.book_id),
                    r.title.replace("\t", " "),
                    r.source_dir.replace("\t", " "),
                    "1" if r.has_overview else "0",
                    "1" if r.has_sections else "0",
                    str(r.chapter_count),
                    str(r.section_count),
                    "1" if r.chapter_summary_ok else "0",
                    r.quality_label.replace("\t", " "),
                    r.recommendation.replace("\t", " "),
                    "1" if r.library_ready else "0",
                    r.notes.replace("\t", " "),
                ]
            )
        )
    p.write_text("\n".join(header + [""] + lines) + "\n", encoding="utf-8")
    return p


def sync_registry(library_root: Path) -> Path:
    rows, meta = compute_registry_rows(Path(library_root))
    return write_registry(Path(library_root), rows, meta)


def read_registry(library_root: Path) -> Tuple[Dict[int, RegistryRow], Dict[str, str]]:
    p = registry_path(Path(library_root))
    if not p.exists():
        return {}, {}
    meta: Dict[str, str] = {}
    rows: Dict[int, RegistryRow] = {}
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            m = line.lstrip("#").strip()
            if ":" in m:
                k, v = m.split(":", 1)
                meta[k.strip()] = v.strip()
            continue
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            book_id = int(parts[0])
        except Exception:
            continue
        rows[book_id] = RegistryRow(
            book_id=book_id,
            title=parts[1],
            source_dir=parts[2],
            has_overview=parts[3] == "1",
            has_sections=parts[4] == "1",
            chapter_count=int(parts[5] or 0),
            section_count=int(parts[6] or 0),
            chapter_summary_ok=parts[7] == "1",
            quality_label=parts[8],
            recommendation=parts[9],
            library_ready=parts[10] == "1",
            notes=parts[11],
        )
    return rows, meta
