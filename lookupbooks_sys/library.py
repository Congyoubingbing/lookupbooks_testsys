from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_read_text(p: Path, encoding: str = "utf-8") -> str:
    try:
        return p.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return p.read_text(encoding=encoding, errors="ignore")


def _load_json(p: Path, default: Any = None) -> Any:
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _norm_title(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff ]+", "", s)
    return s.strip()


def _slugify_title(s: str, max_len: int = 72) -> str:
    t = _norm_title(s).replace(' ', '_')
    t = re.sub(r'_+', '_', t).strip('_')
    if not t:
        t = 'book'
    return t[:max_len].rstrip('_') or 'book'


def _copy_output_tree(src: Path, dst: Path) -> None:
    """Copy only the minimal portable artifacts required by library/.

    Intentionally excludes bulky and rebuildable intermediates such as _cache/,
    images, logs, debug traces, and review artifacts. The embedded book copy only
    keeps the structures and final split texts that the library runtime reads.
    """
    src = Path(src)
    dst = Path(dst)
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    _ensure_dir(dst)

    # Top-level files the portable library may read or that are useful as the
    # canonical, final state for the embedded book package.
    top_level_keep = {
        'book_overview.json',
        'chapter_overview.json',
        'final_book_state.json',
        'README.txt',
        'README.md',
    }

    # Directory payloads actually consumed by library runtime.
    dir_keep = {'chapters', 'sections'}

    for child in src.iterdir():
        if child.name.startswith('.nfs'):
            continue
        target = dst / child.name
        if child.is_file():
            if child.name not in top_level_keep:
                continue
            _ensure_dir(target.parent)
            shutil.copy2(child, target)
            continue
        if not child.is_dir() or child.name not in dir_keep:
            continue
        if child.name == 'chapters':
            shutil.copytree(child, target, copy_function=shutil.copy2, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
            continue
        if child.name == 'sections':
            _ensure_dir(target)
            for ch_dir in sorted([p for p in child.iterdir() if p.is_dir()]):
                dst_ch = target / ch_dir.name
                _ensure_dir(dst_ch)
                idx = ch_dir / 'section_index.json'
                if idx.exists():
                    shutil.copy2(idx, dst_ch / 'section_index.json')
                for fp in sorted(ch_dir.glob('*.txt')):
                    shutil.copy2(fp, dst_ch / fp.name)


def _rewrite_paths_in_obj(obj: Any, *, source_prefix: str) -> tuple[Any, bool]:
    changed = False
    src = str(source_prefix or '').replace('\\', '/').rstrip('/')

    def _rewrite_value(v: Any) -> Any:
        nonlocal changed
        if isinstance(v, str):
            s = v.replace('\\', '/')
            if src and (s == src or s.startswith(src + '/')):
                rel = s[len(src):].lstrip('/')
                changed = True
                return rel or '.'
            return v
        if isinstance(v, list):
            out = []
            local_changed = False
            for item in v:
                nv = _rewrite_value(item)
                if nv is not item:
                    local_changed = True
                out.append(nv)
            if local_changed:
                changed = True
            return out
        if isinstance(v, dict):
            out = {}
            local_changed = False
            for k, item in v.items():
                nv = _rewrite_value(item)
                if nv is not item:
                    local_changed = True
                out[k] = nv
            if local_changed:
                changed = True
            return out
        return v

    new_obj = _rewrite_value(obj)
    return new_obj, changed


def _rewrite_embedded_book_paths(book_dir: Path, original_output_dir: Path) -> None:
    book_dir = Path(book_dir)
    src_prefix = str(Path(original_output_dir).resolve())
    for jp in book_dir.rglob('*.json'):
        data = _load_json(jp, default=None)
        if data is None:
            continue
        new_data, changed = _rewrite_paths_in_obj(data, source_prefix=src_prefix)
        if changed:
            jp.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding='utf-8')

    cov = _load_json(book_dir / 'chapter_overview.json', default=None)
    if isinstance(cov, dict):
        chs = cov.get('chapters')
        if isinstance(chs, list):
            changed = False
            for ch in chs:
                if not isinstance(ch, dict):
                    continue
                ch_no = int(ch.get('chapter_no') or 0)
                section_index_rel = f'sections/ch{ch_no:02d}/section_index.json'
                if (book_dir / section_index_rel).exists() and ch.get('section_index_file') != section_index_rel:
                    ch['section_index_file'] = section_index_rel
                    changed = True
                chapter_glob = sorted((book_dir / 'chapters').glob(f'ch{ch_no:02d}_*.txt')) if (book_dir / 'chapters').exists() else []
                if chapter_glob:
                    chapter_rel = f"chapters/{chapter_glob[0].name}"
                    if ch.get('chapter_file') != chapter_rel:
                        ch['chapter_file'] = chapter_rel
                        changed = True
            if changed:
                (book_dir / 'chapter_overview.json').write_text(json.dumps(cov, ensure_ascii=False, indent=2), encoding='utf-8')

    for idx_path in book_dir.glob('sections/ch*/section_index.json'):
        idx = _load_json(idx_path, default=None)
        if not isinstance(idx, dict):
            continue
        items = idx.get('items')
        if not isinstance(items, list):
            continue
        changed = False
        for item in items:
            if not isinstance(item, dict):
                continue
            file_val = str(item.get('file') or '').replace('\\', '/')
            if file_val:
                base = Path(file_val).name
                if base and item.get('file') != base:
                    item['file'] = base
                    changed = True
        if changed:
            idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding='utf-8')


@dataclass
class BookIndexEntry:
    book_id: int
    title: str
    source_dir: str
    source_key: str
    has_sections: bool = False
    has_overview: bool = False
    chapter_count: int = 0
    section_count: int = 0
    recommendation: str = ""
    quality_label: str = ""
    quality_score: float = 0.0
    updated_at: str = ""
    embedded_in_library: bool = False
    original_output_dir: str = ""


@dataclass
class ChapterMeta:
    book_id: int
    chapter_id: str
    chapter_no: int
    title: str
    summary: str = ""
    keywords: List[str] = None
    chapter_exposure_decision: str = ""
    chapter_recommendation: str = ""
    section_index_file: str = ""
    quality_label: str = ""
    quality_score: float = 0.0

    def __post_init__(self) -> None:
        if self.keywords is None:
            self.keywords = []


@dataclass
class SectionMeta:
    book_id: int
    chapter_no: int
    section_id: str
    title: str
    file: str
    summary: str = ""
    keywords: List[str] = None
    exposure_decision: str = ""
    summary_source_allowed: bool = True
    quality_label: str = ""
    quality_score: float = 0.0
    status: str = ""

    def __post_init__(self) -> None:
        if self.keywords is None:
            self.keywords = []


class BookLibrary:
    """Portable library.

    By default library/build keeps a self-contained copy of each book under
    library/books/bookXXX_slug/, so the library can be moved without requiring the
    original outputs/ tree. Lightweight compatibility files are still generated in
    library/ and library/books/.
    """

    def __init__(self, library_root: Path):
        self.root = Path(library_root).resolve()
        self.books_dir = _ensure_dir(self.root / "books")

    @classmethod
    def load_or_create(cls, library_root: Path) -> "BookLibrary":
        return cls(Path(library_root))

    @property
    def index_path(self) -> Path:
        return self.root / "book_index.json"

    @property
    def book_titles_path(self) -> Path:
        return self.root / "book_titles.txt"

    @property
    def catalog_path(self) -> Path:
        return self.root / "library_catalog.json"

    def summarybook_path(self) -> Path:
        return self.root / "summarybook.txt"

    def chapter_summary_path(self, book_id: int) -> Path:
        return self.books_dir / f"book{int(book_id)}_chapter_summary.txt"

    def resolve_source_dir_value(self, source_dir: str) -> Path:
        p = Path(source_dir)
        if p.is_absolute():
            return p.resolve()
        return (self.root / p).resolve()

    def relative_to_root(self, p: Path) -> str:
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except Exception:
            try:
                return p.relative_to(self.root).as_posix()
            except Exception:
                return str(p)

    def embedded_book_dir(self, book_id: int, title: str) -> Path:
        return self.books_dir / f"book{int(book_id):03d}_{_slugify_title(title)}"

    def load_index(self) -> List[BookIndexEntry]:
        data = _load_json(self.index_path, default={}) or {}
        rows = data.get("books", []) if isinstance(data, dict) else []
        out: List[BookIndexEntry] = []
        for row in rows:
            try:
                out.append(
                    BookIndexEntry(
                        book_id=int(row.get("book_id")),
                        title=str(row.get("title") or ""),
                        source_dir=str(row.get("source_dir") or ""),
                        source_key=str(row.get("source_key") or ""),
                        has_sections=bool(row.get("has_sections", False)),
                        has_overview=bool(row.get("has_overview", False)),
                        chapter_count=int(row.get("chapter_count") or 0),
                        section_count=int(row.get("section_count") or 0),
                        recommendation=str(row.get("recommendation") or ""),
                        quality_label=str(row.get("quality_label") or ""),
                        quality_score=float(row.get("quality_score") or 0.0),
                        updated_at=str(row.get("updated_at") or ""),
                        embedded_in_library=bool(row.get("embedded_in_library", False)),
                        original_output_dir=str(row.get("original_output_dir") or ""),
                    )
                )
            except Exception:
                continue
        out.sort(key=lambda x: x.book_id)
        return out

    def save_index(self, entries: Iterable[BookIndexEntry]) -> None:
        entries = sorted(list(entries), key=lambda x: x.book_id)
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "count": len(entries),
            "books": [asdict(e) for e in entries],
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_entry(self, book_id: int) -> Optional[BookIndexEntry]:
        for e in self.load_index():
            if int(e.book_id) == int(book_id):
                return e
        return None

    def get_entry_by_title(self, title: str) -> Optional[BookIndexEntry]:
        n = _norm_title(title)
        for e in self.load_index():
            if _norm_title(e.title) == n:
                return e
        return None

    def read_summarybook(self) -> str:
        return _safe_read_text(self.summarybook_path()) if self.summarybook_path().exists() else ""

    def read_chapter_summary(self, book_id: int) -> str:
        p = self.chapter_summary_path(book_id)
        return _safe_read_text(p) if p.exists() else ""

    def read_catalog(self) -> Dict[str, Any]:
        return _load_json(self.catalog_path, default={}) or {}

    def _source_dir(self, book_id: int) -> Path:
        e = self.get_entry(book_id)
        if not e:
            raise KeyError(f"book_id not found: {book_id}")
        return self.resolve_source_dir_value(e.source_dir)

    def read_book_overview(self, book_id: int) -> Dict[str, Any]:
        root = self._source_dir(book_id)
        p = root / "book_overview.json"
        return _load_json(p, default={}) or {}

    def read_chapter_overview(self, book_id: int) -> Dict[str, Any]:
        root = self._source_dir(book_id)
        p = root / "chapter_overview.json"
        return _load_json(p, default={}) or {}

    def _chapter_overview_row(self, book_id: int, chapter_no: int) -> Dict[str, Any]:
        ov = self.read_chapter_overview(book_id)
        rows = ov.get("chapters", []) if isinstance(ov, dict) else []
        for row in rows:
            try:
                if int(row.get("chapter_no") or 0) == int(chapter_no):
                    return row if isinstance(row, dict) else {}
            except Exception:
                continue
        return {}

    def _resolve_book_relative_path(self, root: Path, rel: str) -> Path:
        rel_s = str(rel or "").replace("\\", "/").strip()
        if not rel_s:
            return Path()
        rel_p = Path(rel_s)
        if rel_p.is_absolute():
            return rel_p
        return root / rel_p

    def _section_dir(self, book_id: int, chapter_no: int) -> Path:
        return self._section_index_path(book_id, chapter_no).parent

    def list_chapters(self, book_id: int, filter_policy: Optional[Dict[str, Any]] = None) -> List[ChapterMeta]:
        ov = self.read_chapter_overview(book_id)
        rows = ov.get("chapters", []) if isinstance(ov, dict) else []
        out: List[ChapterMeta] = []
        allow_exposure = set((filter_policy or {}).get("prefer_exposure", []))
        if not allow_exposure:
            allow_exposure = {"expose", "caution", "masked", "unknown", ""}
        for row in rows:
            try:
                meta = ChapterMeta(
                    book_id=int(book_id),
                    chapter_id=str(row.get("chapter_id") or f"ch{int(row.get('chapter_no') or 0):02d}"),
                    chapter_no=int(row.get("chapter_no") or 0),
                    title=str(row.get("title") or row.get("chapter_title") or "").strip(),
                    summary=str(row.get("summary") or row.get("chapter_summary") or "").strip(),
                    keywords=list(row.get("keywords") or row.get("chapter_keywords") or []),
                    chapter_exposure_decision=str(row.get("chapter_exposure_decision") or row.get("exposure_decision") or ""),
                    chapter_recommendation=str(row.get("chapter_recommendation") or row.get("recommendation") or ""),
                    section_index_file=str(row.get("section_index_file") or ""),
                    quality_label=str(row.get("quality_label") or ""),
                    quality_score=float(row.get("quality_score") or 0.0),
                )
            except Exception:
                continue
            if meta.chapter_exposure_decision not in allow_exposure:
                continue
            out.append(meta)
        out.sort(key=lambda x: x.chapter_no)
        return out

    def _section_index_path(self, book_id: int, chapter_no: int) -> Path:
        root = self._source_dir(book_id)
        row = self._chapter_overview_row(book_id, chapter_no)
        rel = str(row.get("section_index_file") or "").replace("\\", "/").strip() if isinstance(row, dict) else ""
        if rel:
            p = self._resolve_book_relative_path(root, rel)
            if p.exists():
                return p
        return root / "sections" / f"ch{int(chapter_no):02d}" / "section_index.json"

    def list_sections(self, book_id: int, chapter_no: int, filter_policy: Optional[Dict[str, Any]] = None) -> List[SectionMeta]:
        p = self._section_index_path(book_id, chapter_no)
        data = _load_json(p, default={}) or {}
        rows = data.get("items", []) if isinstance(data, dict) else []
        out: List[SectionMeta] = []
        prefer_exposure = set((filter_policy or {}).get("prefer_exposure", []))
        if not prefer_exposure:
            prefer_exposure = {"expose", "caution", "masked", "unknown", ""}
        allow_masked = bool((filter_policy or {}).get("allow_masked_sections", False))
        min_quality = float((filter_policy or {}).get("min_section_quality_score", 0.0) or 0.0)
        for row in rows:
            try:
                meta = SectionMeta(
                    book_id=int(book_id),
                    chapter_no=int(chapter_no),
                    section_id=str(row.get("section_id") or ""),
                    title=str(row.get("title") or "").strip(),
                    file=Path(str(row.get("file") or "")).name or str(row.get("file") or ""),
                    summary=str(row.get("summary") or "").strip(),
                    keywords=list(row.get("keywords") or []),
                    exposure_decision=str(row.get("exposure_decision") or ""),
                    summary_source_allowed=bool(row.get("summary_source_allowed", True)),
                    quality_label=str(row.get("quality_label") or ""),
                    quality_score=float(row.get("quality_score") or 0.0),
                    status=str(row.get("status") or ""),
                )
            except Exception:
                continue
            if meta.quality_score < min_quality:
                continue
            if meta.exposure_decision == "masked" and not allow_masked:
                continue
            if meta.exposure_decision not in prefer_exposure:
                continue
            out.append(meta)
        out.sort(key=lambda x: x.section_id)
        return out

    def read_section_text(self, book_id: int, chapter_no: int, section_id_or_file: str) -> str:
        ch_dir = self._section_dir(book_id, chapter_no).resolve()
        key = str(section_id_or_file or "").strip()
        p = Path(key)

        def _read_if_exists(candidate: Path) -> str:
            try:
                cp = Path(candidate)
                if cp.exists() and cp.is_file():
                    return _safe_read_text(cp)
            except Exception:
                return ""
            return ""

        # 1) Prefer the current chapter directory and basename, independent of cwd.
        if p.name:
            txt = _read_if_exists(ch_dir / p.name)
            if txt:
                return txt

        # 2) If the caller provides a section id like ch03_s007, resolve within chapter dir.
        m = re.match(r"ch(\d+)_s(\d+)", key, flags=re.I)
        if m and ch_dir.exists():
            secno = int(m.group(2))
            matches = sorted(ch_dir.glob(f"s{secno:03d}_*.txt"))
            if matches:
                return _safe_read_text(matches[0])

        # 3) If a section index row exists, use its normalized basename as canonical target.
        idx = _load_json(self._section_index_path(book_id, chapter_no), default={}) or {}
        rows = idx.get("items", []) if isinstance(idx, dict) else []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row_sid = str(row.get("section_id") or "").strip()
                row_file = Path(str(row.get("file") or "")).name
                if key == row_sid or key == row_file or p.name == row_file:
                    txt = _read_if_exists(ch_dir / row_file)
                    if txt:
                        return txt

        # 4) Accept absolute file paths as a last resort.
        if p.is_absolute():
            txt = _read_if_exists(p)
            if txt:
                return txt

        # 5) Accept cwd-relative paths as a final fallback, but only after basename/chapter-dir resolution.
        txt = _read_if_exists(p)
        if txt:
            return txt

        # 6) Stem match fallback inside current chapter directory.
        if ch_dir.exists():
            for fp in ch_dir.glob("*.txt"):
                if fp.name == key or fp.stem == p.stem:
                    return _safe_read_text(fp)
        return ""

    def build_book_titles_file(self) -> str:
        entries = self.load_index()
        lines = [f"{e.book_id}\t{e.title}" for e in entries]
        content = "\n".join(lines) + ("\n" if lines else "")
        self.book_titles_path.write_text(content, encoding="utf-8")
        return content

    def build_summary_files(self) -> None:
        entries = self.load_index()
        sb_lines: List[str] = []
        for e in entries:
            bo = self.read_book_overview(e.book_id)
            summary = str(bo.get("summary") or bo.get("book_summary") or "").strip()
            if not summary:
                summary = f"{e.title}。"
            sb_lines.append(f"{e.book_id}\t{e.title}\t{summary}")

            ch_lines: List[str] = []
            for ch in self.list_chapters(e.book_id, filter_policy={"prefer_exposure": ["expose", "caution", "masked", "unknown", ""]}):
                summ = ch.summary or ch.title
                ch_lines.append(f"chapter{ch.chapter_no}\t{ch.title}\t{summ}")
            self.chapter_summary_path(e.book_id).write_text("\n".join(ch_lines) + ("\n" if ch_lines else ""), encoding="utf-8")
        self.summarybook_path().write_text("\n".join(sb_lines) + ("\n" if sb_lines else ""), encoding="utf-8")

    def build_library_catalog(self) -> Dict[str, Any]:
        entries = self.load_index()
        catalog: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "count": len(entries),
            "books": [],
        }
        for e in entries:
            bo = self.read_book_overview(e.book_id)
            chs = self.list_chapters(e.book_id, filter_policy={"prefer_exposure": ["expose", "caution", "masked", "unknown", ""]})
            catalog["books"].append(
                {
                    "book_id": e.book_id,
                    "title": e.title,
                    "source_dir": e.source_dir,
                    "source_key": e.source_key,
                    "embedded_in_library": e.embedded_in_library,
                    "original_output_dir": e.original_output_dir,
                    "summary": str(bo.get("summary") or ""),
                    "keywords": list(bo.get("keywords") or []),
                    "topics": list(bo.get("topics") or []),
                    "recommendation": e.recommendation,
                    "quality_label": e.quality_label,
                    "quality_score": e.quality_score,
                    "chapter_count": e.chapter_count,
                    "section_count": e.section_count,
                    "chapters": [
                        {
                            "chapter_id": c.chapter_id,
                            "chapter_no": c.chapter_no,
                            "title": c.title,
                            "summary": c.summary,
                            "keywords": c.keywords,
                            "chapter_exposure_decision": c.chapter_exposure_decision,
                            "chapter_recommendation": c.chapter_recommendation,
                            "section_index_file": c.section_index_file,
                            "quality_label": c.quality_label,
                            "quality_score": c.quality_score,
                        }
                        for c in chs
                    ],
                }
            )
        self.catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
        return catalog


def _book_output_valid(book_dir: Path) -> bool:
    return book_dir.is_dir() and (
        (book_dir / "book_overview.json").exists()
        or (book_dir / "chapter_overview.json").exists()
        or (book_dir / "sections").exists()
        or (book_dir / "chapters").exists()
    )


def _build_entry_from_output_dir(book_dir: Path, book_id: int, *, source_dir: Optional[str] = None, embedded_in_library: bool = False, original_output_dir: Optional[str] = None) -> BookIndexEntry:
    bo = _load_json(book_dir / "book_overview.json", default={}) or {}
    fs = _load_json(book_dir / "final_book_state.json", default={}) or {}
    cov = _load_json(book_dir / "chapter_overview.json", default={}) or {}
    title = str(bo.get("book_title") or cov.get("book_title") or book_dir.name)
    chapters = cov.get("chapters", []) if isinstance(cov, dict) else []
    chapter_count = int(bo.get("chapter_count") or len(chapters) or 0)
    section_count = int(bo.get("section_count") or bo.get("usable_section_count") or 0)
    if not section_count:
        # compute from indexes
        sections_root = book_dir / "sections"
        if sections_root.exists():
            count = 0
            for p in sections_root.glob("ch*/section_index.json"):
                idx = _load_json(p, default={}) or {}
                count += len(idx.get("items", [])) if isinstance(idx, dict) else 0
            section_count = count
    q = fs.get("quality") or {}
    return BookIndexEntry(
        book_id=int(book_id),
        title=title,
        source_dir=str(source_dir or book_dir.resolve()),
        source_key=book_dir.name,
        has_sections=(book_dir / "sections").exists(),
        has_overview=(book_dir / "book_overview.json").exists() and (book_dir / "chapter_overview.json").exists(),
        chapter_count=chapter_count,
        section_count=section_count,
        recommendation=str(bo.get("recommendation") or q.get("recommendation") or ""),
        quality_label=str(bo.get("quality_label") or q.get("quality_label") or ""),
        quality_score=float(bo.get("quality_score") or q.get("quality_score") or 0.0),
        updated_at=datetime.now().isoformat(timespec="seconds"),
        embedded_in_library=bool(embedded_in_library),
        original_output_dir=str(original_output_dir or book_dir.resolve()),
    )


def import_from_pdf_txt_align_outputs(
    outputs_root: Path,
    library_root: Path,
    *,
    start_book_id: int = 1,
    mode: str = "manifest",
    overwrite: bool = False,
    only_updated_from: Optional[Path] = None,
    embed_artifacts: bool = True,
) -> List[BookIndexEntry]:
    outputs_root = Path(outputs_root)
    lib = BookLibrary(Path(library_root))
    existing = lib.load_index()
    by_norm_title = {_norm_title(e.title): e for e in existing}
    by_id = {e.book_id: e for e in existing}
    used_ids = set(by_id.keys())

    def next_id(cur: int) -> int:
        while cur in used_ids:
            cur += 1
        used_ids.add(cur)
        return cur

    selected_names: Optional[set] = None
    if only_updated_from:
        selected_names = set()
        for pdf in sorted(Path(only_updated_from).glob("*.pdf")):
            selected_names.add(_norm_title(pdf.stem))

    for book_dir in sorted([p for p in outputs_root.iterdir() if _book_output_valid(p)], key=lambda p: p.name.lower()):
        probe = _build_entry_from_output_dir(book_dir, book_id=0)
        title_key = _norm_title(probe.title)
        if selected_names is not None and title_key not in selected_names and _norm_title(book_dir.name) not in selected_names:
            continue
        ex = by_norm_title.get(title_key)
        if ex and not overwrite:
            if not (embed_artifacts and not bool(getattr(ex, "embedded_in_library", False))):
                continue
        book_id = ex.book_id if ex else next_id(start_book_id)
        start_book_id = max(start_book_id, book_id + 1)

        target_source_dir = str(book_dir.resolve())
        embedded = False
        if embed_artifacts:
            if ex and ex.embedded_in_library and ex.source_dir:
                target_book_dir = lib.resolve_source_dir_value(ex.source_dir)
            else:
                target_book_dir = lib.embedded_book_dir(book_id, probe.title)
            _copy_output_tree(book_dir, target_book_dir)
            _rewrite_embedded_book_paths(target_book_dir, book_dir)
            target_source_dir = lib.relative_to_root(target_book_dir)
            embedded = True

        ent = _build_entry_from_output_dir(
            book_dir,
            book_id=book_id,
            source_dir=target_source_dir,
            embedded_in_library=embedded,
            original_output_dir=str(book_dir.resolve()),
        )
        by_id[book_id] = ent
        by_norm_title[title_key] = ent

    entries = sorted(by_id.values(), key=lambda x: x.book_id)
    lib.save_index(entries)
    lib.build_book_titles_file()
    lib.build_summary_files()
    lib.build_library_catalog()
    return entries


def add_book_from_output_dir(
    book_output_dir: Path,
    library_root: Path,
    *,
    book_id: Optional[int] = None,
    title: Optional[str] = None,
    mode: str = "manifest",
    overwrite: bool = False,
    embed_artifacts: bool = True,
) -> BookIndexEntry:
    book_output_dir = Path(book_output_dir)
    if not _book_output_valid(book_output_dir):
        raise FileNotFoundError(f"book_output_dir not found or invalid: {book_output_dir}")
    lib = BookLibrary(Path(library_root))
    entries = lib.load_index()
    used_ids = {e.book_id for e in entries}
    if book_id is None:
        book_id = 1
        while book_id in used_ids:
            book_id += 1
    probe = _build_entry_from_output_dir(book_output_dir, book_id=book_id)
    if title:
        probe.title = title
    target_source_dir = str(book_output_dir.resolve())
    embedded = False
    if embed_artifacts:
        existing = next((e for e in entries if e.book_id == book_id), None)
        if existing and existing.embedded_in_library and existing.source_dir:
            target_book_dir = lib.resolve_source_dir_value(existing.source_dir)
        else:
            target_book_dir = lib.embedded_book_dir(book_id, probe.title)
        _copy_output_tree(book_output_dir, target_book_dir)
        _rewrite_embedded_book_paths(target_book_dir, book_output_dir)
        target_source_dir = lib.relative_to_root(target_book_dir)
        embedded = True
    ent = _build_entry_from_output_dir(
        book_output_dir,
        book_id=book_id,
        source_dir=target_source_dir,
        embedded_in_library=embedded,
        original_output_dir=str(book_output_dir.resolve()),
    )
    if title:
        ent.title = title
    replaced = False
    new_entries: List[BookIndexEntry] = []
    for e in entries:
        if e.book_id == ent.book_id:
            if not overwrite:
                raise ValueError(f"book_id already exists: {ent.book_id}; pass overwrite=True")
            new_entries.append(ent)
            replaced = True
        else:
            new_entries.append(e)
    if not replaced:
        new_entries.append(ent)
    new_entries.sort(key=lambda x: x.book_id)
    lib.save_index(new_entries)
    lib.build_book_titles_file()
    lib.build_summary_files()
    lib.build_library_catalog()
    return ent


def update_book_from_output_dir(
    library_root: Path,
    *,
    book_id: int,
    book_output_dir: Path,
    overwrite: bool = True,
    embed_artifacts: bool = True,
) -> BookIndexEntry:
    lib = BookLibrary(Path(library_root))
    if lib.get_entry(book_id) is None:
        raise KeyError(f"book_id not found: {book_id}")
    return add_book_from_output_dir(book_output_dir, library_root, book_id=book_id, overwrite=overwrite, embed_artifacts=embed_artifacts)


def remove_book_from_library(library_root: Path, book_id: int, *, delete_files: bool = False) -> bool:
    lib = BookLibrary(Path(library_root))
    entries = lib.load_index()
    target = None
    kept: List[BookIndexEntry] = []
    for e in entries:
        if int(e.book_id) == int(book_id):
            target = e
        else:
            kept.append(e)
    if target is None:
        return False
    lib.save_index(kept)
    try:
        lib.chapter_summary_path(book_id).unlink(missing_ok=True)
    except Exception:
        pass
    lib.build_book_titles_file()
    lib.build_summary_files()
    lib.build_library_catalog()
    if delete_files and target.source_dir:
        try:
            shutil.rmtree(lib.resolve_source_dir_value(target.source_dir), ignore_errors=True)
        except Exception:
            pass
    return True
