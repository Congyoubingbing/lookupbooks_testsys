from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import inspect
import re

from .pdf_units import PDFUnitStore
from .llm_calls import vl_score_is_toc, vl_extract_toc_markdown, llm_parse_toc
from .utils import dump_json, load_json

TOC_CACHE_VERSION = 8


# Separate cache version for TOC scan results (toc_scan.json). Bump if scan logic/schema changes.
TOC_SCAN_CACHE_VERSION = 3


def _min_required_chapters(cfg, unit_count: Optional[int] = None) -> int:
    """Compute an adaptive minimum chapter-count threshold.

    We keep strong global-TOC checks, but do not hard-fail otherwise valid small
    textbooks with only ~5 top-level chapters.

    Config:
      - toc.min_required_chapters: default threshold (legacy)
      - toc.min_required_chapters_hard: absolute lower bound
      - toc.smallbook_max_units / toc.smallbook_min_required_chapters: override
        for small books (unit_count <= max_units)
    """
    base = int(getattr(getattr(cfg, "toc", None), "min_required_chapters", 6) or 6)
    hard = int(getattr(getattr(cfg, "toc", None), "min_required_chapters_hard", min(base, 4)) or min(base, 4))
    hard = max(2, min(base, hard))

    sb_max = int(getattr(getattr(cfg, "toc", None), "smallbook_max_units", 0) or 0)
    sb_min = int(getattr(getattr(cfg, "toc", None), "smallbook_min_required_chapters", 0) or 0)

    if unit_count is not None and sb_max > 0 and unit_count <= sb_max and sb_min > 0:
        return max(hard, min(base, sb_min))
    return base

def _is_template_title(title: str) -> bool:
    """Return True if title is a bare structural template (e.g., 'Chapter 3', '第3章') with no semantic words."""
    if not title:
        return True
    t = str(title).strip()
    # normalize spaces
    t = re.sub(r"\s+", " ", t)
    tl = t.lower()

    # English templates
    if re.fullmatch(r"(chapter|chap)\s*[0-9]{1,4}", tl):
        return True
    if re.fullmatch(r"part\s*(?:[0-9]{1,3}|[ivxlcdm]{1,8})", tl):
        return True
    if re.fullmatch(r"(appendix|app\.)\s*(?:[a-z]|[0-9]{1,3}|[ivxlcdm]{1,8})", tl):
        return True

    # Chinese templates
    if re.fullmatch(r"第\s*[0-9]{1,4}\s*章", t):
        return True
    if re.fullmatch(r"第\s*[一二三四五六七八九十百千零〇]{1,6}\s*章", t):
        return True

    return False

# --- Chinese numeral helpers (for titles like '第十章') ---
_CN_DIGITS = {
    "零": 0, "〇": 0, "○": 0,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}

def _cn_numeral_to_int(s: str) -> Optional[int]:
    """Convert common Chinese numerals to int (supports up to 9999)."""
    if not s:
        return None
    s = re.sub(r"\s+", "", s)
    # fast path: ascii/fullwidth digits
    s_norm = s.translate(str.maketrans({"０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9"}))
    if s_norm.isdigit():
        try:
            return int(s_norm)
        except Exception:
            return None
    total = 0
    num = 0
    unit_seen = False
    for ch in s:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            unit_seen = True
            if num == 0:
                num = 1
            total += num * unit
            num = 0
        else:
            # ignore other chars
            continue
    total += num
    if total == 0 and unit_seen:
        # e.g., "十"
        total = 10
    return total if total > 0 else None

def _extract_chapter_no_from_title(title: str) -> Optional[int]:
    """Extract chapter number from title like '第十章 ...' or '第 3 章 ...'."""
    t = (title or "").strip()
    m = re.match(r"^第\s*([0-9０-９]{1,3}|[零〇○一二两三四五六七八九十百千]{1,8})\s*章", t)
    if not m:
        return None
    raw = m.group(1)
    n = _cn_numeral_to_int(raw)
    return n

# ------------------------------
# TOC scan (image + VL classifier)
# ------------------------------

@dataclass
class TOCRange:
    start_unit: int
    end_unit: int


def _resolve_vision_model(cfg) -> str:
    return str(getattr(cfg.models, "vision_model", "") or getattr(cfg.models, "vl_model", "") or "qwen3-vl-plus")


def _resolve_llm_model(cfg) -> str:
    return str(getattr(cfg.models, "llm_model", "") or getattr(cfg.models, "model", "") or "qwen3.5-397b-a17b")


def _heuristic_toc_score(text: str) -> float:
    """A cheap heuristic to detect BOOK-LEVEL (global) TOC-like pages from extracted text.

    Key goal: avoid false positives on "chapter outline / chapter contents" pages where
    most numbered entries share the same top-level prefix (e.g., all start with '1.').
    """
    t = (text or "")
    if not t:
        return 0.0
    tl = t.lower()
    score = 0.0

    has_kw = ("contents" in tl) or ("table of contents" in tl) or ("目 录" in tl) or ("目录" in tl)
    if has_kw:
        score += 0.45

    lines = [ln.strip() for ln in tl.splitlines() if ln.strip()]
    if not lines:
        return max(0.0, min(1.0, score))

    tail_num = 0
    dots = 0
    top_prefixes: List[int] = []
    max_page = 0

    for ln in lines[:160]:
        m_end = re.search(r"(\d{1,4})\s*$", ln)
        if m_end:
            tail_num += 1
            try:
                max_page = max(max_page, int(m_end.group(1)))
            except Exception:
                pass
        if "...." in ln or "…" in ln or "·" in ln:
            dots += 1

        # top-level numeric prefix like 3.2.1 -> 3
        m = re.match(r"^\s*(\d{1,3})\s*\.\s*\d", ln)
        if m:
            try:
                top_prefixes.append(int(m.group(1)))
            except Exception:
                pass
        # Chapter 3 / 第3章
        m2 = re.match(r"^\s*chapter\s+(\d{1,3})\b", ln)
        if m2:
            try:
                top_prefixes.append(int(m2.group(1)))
            except Exception:
                pass
        m3 = re.match(r"^\s*第\s*(\d{1,3})\s*章", ln)
        if m3:
            try:
                top_prefixes.append(int(m3.group(1)))
            except Exception:
                pass
        # Chinese numeral chapter prefix: 第十章/第一章
        m4 = re.match(r"^\s*第\s*([一二三四五六七八九十百千零〇两]{1,6})\s*章", ln)
        if m4:
            n = _cn_numeral_to_int(m4.group(1))
            if isinstance(n, int):
                top_prefixes.append(int(n))

    denom = max(1, min(len(lines), 160))
    ratio_tail = tail_num / denom
    ratio_dots = dots / denom
    score += min(0.45, ratio_tail * 1.2)
    score += min(0.18, ratio_dots * 0.5)

    # Penalize "single-prefix section list" (chapter outline)
    uniq_prefix = len(set(top_prefixes)) if top_prefixes else 0
    if uniq_prefix <= 1 and tail_num >= 6:
        score -= 0.25
        if max_page and max_page <= 35:
            score -= 0.15

    return max(0.0, min(1.0, score))


def find_toc_range(
    store: PDFUnitStore,
    session_vl,
    cfg,
    cache_dir: Path,
    logger: logging.Logger,
) -> Tuple[TOCRange, Dict[str, Any]]:
    """Find a unit range likely containing the BOOK-LEVEL TOC.

    Writes:
      - toc_scan.json
    """
    cache_path = cache_dir / "toc_scan.json"
    cached = load_json(cache_path)
    if cached and isinstance(cached, dict) and "range" in cached:
        try:
            if int(cached.get("_cache_version", 0)) == int(TOC_SCAN_CACHE_VERSION):
                r = cached["range"]
                return TOCRange(int(r[0]), int(r[1])), cached
        except Exception:
            pass

    max1 = int(getattr(cfg.pdf, "toc_scan_max_units_pass1", 260) or 260)
    max2 = int(getattr(cfg.pdf, "toc_scan_max_units_pass2", 220) or 220)
    max3 = int(getattr(cfg.pdf, "toc_scan_tail_units_pass3", 90) or 90)
    stride2 = int(getattr(cfg.pdf, "toc_scan_stride_pass2", 8) or 8)

    score_th = float(getattr(cfg.pdf, "toc_scan_score_threshold", 0.62) or 0.62)
    bwd = int(getattr(cfg.pdf, "toc_range_backward_units", 30) or 30)
    fwd = int(getattr(cfg.pdf, "toc_range_forward_units", 70) or 70)

    # Confirmation / anti-false-positive (chapter outline vs global TOC)
    confirm_global = bool(getattr(cfg.toc, "scan_confirm_global", True))
    confirm_max_vl_pages = int(getattr(cfg.toc, "scan_confirm_max_vl_pages", 1) or 1)
    # Rendering DPI used for TOC scoring vs markdown extraction (VL).
    dpi_score = int(getattr(getattr(cfg, "pdf", None), "dpi_low", 120) or 120)
    dpi_extract = int(getattr(getattr(cfg, "pdf", None), "dpi_high", 240) or dpi_score)


    report: Dict[str, Any] = {
        "pass1": [],
        "pass2": [],
        "pass3": [],
        "fallbacks": [],
        "threshold": score_th,
        "_cache_version": int(TOC_SCAN_CACHE_VERSION),
    }

    def _is_global_from_stats(stats: Dict[str, Any]) -> bool:
        if not stats:
            return False
        distinct = int(stats.get("distinct_chapter_prefixes") or 0)
        max_page = int(stats.get("max_page_num") or 0)
        has_kw = bool(stats.get("has_contents_keyword"))
        min_distinct = int(getattr(cfg.toc, "global_min_distinct_prefixes", 3) or 3)
        min_max_page = int(getattr(cfg.toc, "global_min_max_page", 50) or 50)
        # allow unnumbered TOCs if explicit contents keyword + strong trailing page numbers
        return (distinct >= min_distinct) or (has_kw and max_page >= min_max_page)

    def confirm_hit(unit_idx: int) -> Tuple[bool, Dict[str, Any]]:
        """Confirm candidate is a BOOK-level TOC (not chapter outline)."""
        unit = store.unit_ref(int(unit_idx))
        try:
            text = (store.extract_unit_text(unit, region="full") or "").strip()
        except Exception:
            text = ""
        if text:
            md = filter_toc_markdown(text, cfg)
            st = _toc_markdown_stats(md, cfg)
            return _is_global_from_stats(st), st

        if not session_vl or confirm_max_vl_pages <= 0:
            return False, {"reason": "no_text_no_vl"}

        try:
            # Confirm with a small multi-page window: many TOCs span 2 pages.
            images = []
            for k in range(max(1, int(confirm_max_vl_pages))):
                ui2 = int(unit_idx) + k
                if ui2 > (store.unit_count - 1):
                    break
                try:
                    images.append(store.render_unit(store.unit_ref(int(ui2)), dpi=dpi_extract, region="full"))
                except Exception:
                    continue
            md = vl_extract_toc_markdown(
                session_vl,
                images,
                model=_resolve_vision_model(cfg),
                enable_thinking=bool(getattr(cfg.models, "vision_enable_thinking", False)),
            )
            md = filter_toc_markdown((md or "").strip(), cfg)
            st = _toc_markdown_stats(md, cfg)
            st["confirm_units"] = [int(unit_idx) + i for i in range(len(images))]
            # also require basic markdown validity to reduce random OCR noise
            ok_basic, _ = validate_toc_markdown(md, cfg)
            return bool(ok_basic and _is_global_from_stats(st)), st
        except Exception as e:
            return False, {"reason": "vl_confirm_error", "error": str(e)}

    def score_unit(unit_idx: int) -> float:
        # 1) text-layer heuristic
        try:
            unit = store.unit_ref(int(unit_idx))
            text = (store.extract_unit_text(unit, region="full") or "").strip()
        except Exception:
            text = ""
        s = _heuristic_toc_score(text) if text else 0.0
        if s >= score_th:
            return float(s)

        # 2) VL classifier (scanned / empty text)
        if not session_vl:
            return float(s)
        try:
            img = store.render_unit(store.unit_ref(int(unit_idx)), dpi=dpi_score, region="full")
            out = vl_score_is_toc(
                session_vl,
                img,
                model=_resolve_vision_model(cfg),
                enable_thinking=bool(getattr(cfg.models, "vision_enable_thinking", False)),
            )
            s2 = float(out.get("score", 0.0) or 0.0)
            return max(float(s), min(1.0, s2))
        except Exception:
            return float(s)

    def scan_units(unit_indices: List[int], pass_name: str, max_units: int) -> Optional[int]:
        count = 0
        for ui in unit_indices:
            if count >= max_units:
                break
            s = score_unit(ui)
            rec = {"unit": int(ui), "score": round(float(s), 4)}
            report[pass_name].append(rec)
            count += 1
            if float(s) >= score_th:
                if not confirm_global:
                    report["hit_unit"] = int(ui)
                    report["hit_score"] = round(float(s), 4)
                    return int(ui)
                ok, st = confirm_hit(int(ui))
                rec["confirmed_global"] = bool(ok)
                rec["confirm_stats"] = st
                if ok:
                    report["hit_unit"] = int(ui)
                    report["hit_score"] = round(float(s), 4)
                    return int(ui)
        return None

    # Pass 1: dense scan from front
    units1 = list(range(0, min(store.unit_count, max1)))
    hit = scan_units(units1, "pass1", max1)

    # Pass 2: stride scan (middle)
    if hit is None:
        units2 = list(range(0, min(store.unit_count, max2), max(1, stride2)))
        hit = scan_units(units2, "pass2", max2)

    # Pass 3: tail scan (back)
    if hit is None:
        tail_start = max(0, store.unit_count - max3)
        units3 = list(range(tail_start, store.unit_count))
        hit = scan_units(units3, "pass3", max3)

    if hit is None:
        # fallback: assume TOC near front
        report["fallbacks"].append("toc_not_found_assume_front")
        hit = 0

    start = max(0, hit - bwd)
    end = min(store.unit_count - 1, hit + fwd)
    report["range"] = [start, end]
    report["hit_unit"] = int(hit)
    dump_json(cache_path, report)
    return TOCRange(start, end), report


# ------------------------------
# TOC markdown extraction + JSON parse
# ------------------------------



def _get_scan_anchor_units(cache_dir: Path, cfg, toc_range: TOCRange) -> List[int]:
    """Derive top-K anchor units from toc_scan.json for better extraction windowing."""
    scan = load_json(cache_dir / "toc_scan.json") or {}
    candidates: List[Tuple[float, int]] = []
    for k in ("pass1", "pass2", "pass3"):
        for rec in (scan.get(k) or []):
            if not isinstance(rec, dict):
                continue
            try:
                ui = int(rec.get("unit"))
                sc = float(rec.get("score", 0.0) or 0.0)
            except Exception:
                continue
            if ui < int(toc_range.start_unit) or ui > int(toc_range.end_unit):
                continue
            candidates.append((sc, ui))
    if not candidates:
        return []
    # keep topK by score (dedupe)
    topk = int(getattr(cfg.toc, "anchor_topk", 3) or 3)
    min_score = float(getattr(cfg.toc, "anchor_min_score", 0.20) or 0.20)
    seen = set()
    anchors: List[int] = []
    for sc, ui in sorted(candidates, key=lambda x: x[0], reverse=True):
        if sc < min_score:
            continue
        if ui in seen:
            continue
        anchors.append(ui)
        seen.add(ui)
        if len(anchors) >= topk:
            break
    return anchors


def _select_toc_units_for_extraction(
    store: PDFUnitStore,
    toc_range: TOCRange,
    cfg,
    max_units: int = 12,
    anchors: Optional[List[int]] = None,
) -> List[int]:
    """Pick a small set of units to feed TOC extraction.

    - If anchors are provided (from toc_scan), prefer a small ordered neighborhood around
      anchors. The ordering is forward-first then backward (to capture multi-page TOCs),
      rather than slicing the earliest pages which can accidentally pull in cover/foreword.
    - Otherwise, fall back to text-layer heuristic scoring within the range.
    """
    start = int(toc_range.start_unit)
    end = int(toc_range.end_unit)
    if start > end:
        start, end = end, start
    start = max(0, start)
    end = min(store.unit_count - 1, end)

    width = end - start + 1
    if width <= max_units:
        return list(range(start, end + 1))

    # Anchor-window strategy (works best for scanned PDFs where text layer is empty)
    anchors = [int(a) for a in (anchors or []) if isinstance(a, (int, float, str)) and str(a).strip().isdigit()]
    anchors = [a for a in anchors if start <= a <= end]
    if anchors:
        back = int(getattr(cfg.toc, "anchor_window_backward_units", 10) or 10)
        fwd = int(getattr(cfg.toc, "anchor_window_forward_units", 4) or 4)
        selected: List[int] = []

        def _push(ui: int):
            if ui < start or ui > end:
                return
            if ui in seen_local:
                return
            seen_local.add(ui)
            selected.append(int(ui))

        # Build an ordered list around anchors: anchor page, then forward pages, then backward pages.
        seen_local = set()
        for a in anchors:
            a = int(a)
            _push(a)
            for k in range(1, fwd + 1):
                _push(a + k)
            for k in range(1, back + 1):
                _push(a - k)

        if len(selected) > max_units:
            selected = selected[:max_units]
        if selected:
            return selected

    # Text-layer heuristic scoring
    scored: List[Tuple[float, int]] = []
    for ui in range(start, end + 1):
        unit = store.unit_ref(int(ui))
        text = (store.extract_unit_text(unit, region="full") or "").strip()
        if not text:
            continue
        s = _heuristic_toc_score(text)
        if s > 0.0:
            scored.append((float(s), ui))
    if not scored:
        # fall back to early contiguous window
        return list(range(start, start + max_units))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Expand around best-scoring unit, biased backward
    best = scored[0][1]
    back = int(getattr(cfg.toc, "anchor_window_backward_units", 10) or 10)
    fwd = int(getattr(cfg.toc, "anchor_window_forward_units", 4) or 4)
    lo = max(start, best - back)
    hi = min(end, best + fwd)
    # Ordered neighborhood around best: forward-first then backward, to avoid slicing the earliest pages.
    units: List[int] = []
    seen_local = set()
    def _push(ui: int):
        if ui < lo or ui > hi:
            return
        if ui in seen_local:
            return
        seen_local.add(ui)
        units.append(int(ui))
    _push(best)
    for k in range(1, fwd + 1):
        _push(best + k)
    for k in range(1, back + 1):
        _push(best - k)
    if len(units) > max_units:
        units = units[:max_units]
    return units


def _extract_toc_markdown_text_layer(
    store: PDFUnitStore,
    toc_range: TOCRange,
    cfg,
    anchors: Optional[List[int]] = None,
) -> str:
    # When the scan range is wide, concatenate only a small set of high-likelihood units.
    max_units = int(getattr(cfg.toc, "text_max_units", 12) or 12)
    units = _select_toc_units_for_extraction(store, toc_range, cfg, max_units=max_units, anchors=anchors)
    lines: List[str] = []
    for ui in units:
        try:
            unit = store.unit_ref(int(ui))
            txt = store.extract_unit_text(unit, region="full")
            if txt:
                lines.append(txt)
        except Exception:
            continue
    md = "\n".join(lines)
    # Keep line breaks; do not aggressively normalize here.
    return md.strip()

def _extract_toc_markdown_vl(store: PDFUnitStore, session_vl, cfg, toc_range: TOCRange, anchors: Optional[List[int]] = None) -> str:
    # IMPORTANT: limit number of pages fed to VL; large batches often degrade into
    # numeric-only outputs (page-number column) and amplify TOC confusion.
    max_units = int(getattr(cfg.toc, "vl_max_units", 6) or 6)
    units = _select_toc_units_for_extraction(store, toc_range, cfg, max_units=max_units, anchors=anchors)

    images = []
    for ui in units:
        try:
            unit = store.unit_ref(int(ui))
            img = store.render_unit(unit, dpi=store.dpi_high, region="full")
            images.append(img)
        except Exception:
            continue
    if not images:
        return ""
    md = vl_extract_toc_markdown(
        session_vl,
        images,
        model=_resolve_vision_model(cfg),
        enable_thinking=bool(getattr(cfg.models, "vision_enable_thinking", False)),
    )
    return (md or "").strip()


def _toc_markdown_stats(md: str, cfg) -> Dict[str, Any]:
    md = (md or "").strip()
    lines = [ln.rstrip() for ln in md.splitlines() if ln.strip()]
    alpha_lines = sum(1 for ln in lines if re.search(r"[A-Za-z\u4e00-\u9fff]", ln))
    tail_digit_lines = sum(1 for ln in lines if re.search(r"\d\s*$", ln))
    dotleader_lines = sum(1 for ln in lines if ("...." in ln) or ("…" in ln) or ("·" in ln))
    entry_like_lines = sum(
        1 for ln in lines
        if re.search(r"[A-Za-z\u4e00-\u9fff]", ln)
        and (re.search(r"\d\s*$", ln) or ("...." in ln) or ("…" in ln) or ("·" in ln))
    )
    has_contents_keyword = bool(re.search(r"\bcontents\b|\btable of contents\b|目录|目\s*录", md, flags=re.IGNORECASE))

    # Global-TOC signals: distinct top-level chapter prefixes & max page number.
    def roman_to_int(s: str) -> Optional[int]:
        s = (s or "").strip().lower()
        if not s or not re.fullmatch(r"[ivxlcdm]{1,10}", s):
            return None
        vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
        total = 0
        prev = 0
        for ch in reversed(s):
            v = vals.get(ch, 0)
            if v < prev:
                total -= v
            else:
                total += v
                prev = v
        return total if total > 0 else None

    chapter_prefixes: List[int] = []
    max_page_num = 0
    for ln in lines[:200]:
        # trailing page number
        m_end = re.search(r"(\d{1,4})\s*$", ln)
        if m_end:
            try:
                max_page_num = max(max_page_num, int(m_end.group(1)))
            except Exception:
                pass

        s = ln.strip()
        # 第3章
        m = re.match(r"^第\s*([0-9]{1,3})\s*章", s)
        if m:
            try:
                chapter_prefixes.append(int(m.group(1)))
            except Exception:
                pass
            continue
        # 第十章/第一章
        m = re.match(r"^第\s*([一二三四五六七八九十百千零〇两]{1,6})\s*章", s)
        if m:
            n = _cn_numeral_to_int(m.group(1))
            if isinstance(n, int) and n > 0:
                chapter_prefixes.append(int(n))
            continue
        # CHAPTER 3 / Chap. 3 / Chapter III
        m = re.match(r"^(?:chapter|chap\.?)\s+([0-9]{1,3}|[ivxlcdm]{1,10})\b", s, flags=re.IGNORECASE)
        if m:
            tok = m.group(1)
            if tok.isdigit():
                chapter_prefixes.append(int(tok))
            else:
                v = roman_to_int(tok)
                if v is not None:
                    chapter_prefixes.append(v)
            continue
        # 3.2.1 -> 3
        m = re.match(r"^([0-9]{1,3})\s*\.\s*\d", s)
        if m:
            try:
                chapter_prefixes.append(int(m.group(1)))
            except Exception:
                pass
            continue
        # 3 Introduction
        m = re.match(r"^([0-9]{1,3})\s+[A-Za-z\u4e00-\u9fff]", s)
        if m and not re.match(r"^[0-9]{1,3}\s*\.\s*\d", s):
            try:
                chapter_prefixes.append(int(m.group(1)))
            except Exception:
                pass
            continue

    distinct_chapter_prefixes = len(set(chapter_prefixes)) if chapter_prefixes else 0
    min_chapter_prefix = min(chapter_prefixes) if chapter_prefixes else None

    return {
        "md_len": len(md),
        "num_lines": len(lines),
        "alpha_lines": alpha_lines,
        "tail_digit_lines": tail_digit_lines,
        "dotleader_lines": dotleader_lines,
        "entry_like_lines": entry_like_lines,
        "has_contents_keyword": has_contents_keyword,
        "distinct_chapter_prefixes": distinct_chapter_prefixes,
        "min_chapter_prefix": min_chapter_prefix,
        "max_page_num": max_page_num,
    }


def validate_toc_markdown(md: str, cfg) -> Tuple[bool, Dict[str, Any]]:
    md = (md or "").strip()
    st = _toc_markdown_stats(md, cfg)

    min_len = int(getattr(cfg.toc, "min_markdown_len", 220) or 220)
    if st["md_len"] < min_len:
        st["reason"] = "too_short"
        return False, st

    min_lines = int(getattr(cfg.toc, "min_lines", 12) or 12)
    if st["num_lines"] < min_lines:
        st["reason"] = "too_few_lines"
        return False, st

    # Need enough "entry-like" lines: alphabetic + trailing digits
    alpha_min = int(getattr(cfg.toc, "min_alpha_lines", 6) or 6)
    if st["alpha_lines"] < alpha_min:
        st["reason"] = "too_few_alpha_lines"
        return False, st

    tail_min = int(getattr(cfg.toc, "min_tail_digit_lines", 6) or 6)
    if st["tail_digit_lines"] < tail_min:
        # Some TOCs are multi-column or have split page numbers; allow a weak pass when other
        # strong TOC signals exist (dot leaders / many entry-like lines / distinct prefixes).
        dot_min = int(getattr(cfg.toc, "weak_tail_digit_dotleader_min", 3) or 3)
        entry_like_min = int(getattr(cfg.toc, "weak_tail_digit_entrylike_min", max(6, tail_min)) or max(6, tail_min))
        prefix_min = int(getattr(cfg.toc, "weak_tail_digit_prefix_min", 3) or 3)
        dot_ok = int(st.get("dotleader_lines", 0) or 0) >= dot_min
        entry_ok = int(st.get("entry_like_lines", 0) or 0) >= entry_like_min
        prefix_ok = int(st.get("distinct_chapter_prefixes", 0) or 0) >= prefix_min
        if dot_ok or (entry_ok and prefix_ok):
            st["warn"] = "weak_tail_digit_lines"
        else:
            st["reason"] = "too_few_tail_digit_lines"
            return False, st

    # Global-TOC gating (avoid chapter outline pages)
    if bool(getattr(cfg.toc, "require_global_toc", True)):
        min_distinct = int(getattr(cfg.toc, "global_min_distinct_prefixes", 3) or 3)
        min_max_page = int(getattr(cfg.toc, "global_min_max_page", 50) or 50)

        distinct = int(st.get("distinct_chapter_prefixes") or 0)
        max_page = int(st.get("max_page_num") or 0)
        has_kw = bool(st.get("has_contents_keyword"))

        ok_global = (distinct >= min_distinct) or (has_kw and max_page >= min_max_page)
        if not ok_global:
            st["reason"] = "not_global_toc"
            return False, st

        if bool(getattr(cfg.toc, "require_starts_near_one", True)):
            start_max = int(getattr(cfg.toc, "start_near_one_max", 2) or 2)
            min_pref = st.get("min_chapter_prefix")
            if isinstance(min_pref, int) and min_pref > start_max:
                st["reason"] = "toc_starts_late"
                return False, st

    st["reason"] = "ok"
    return True, st


def filter_toc_markdown(md: str, cfg) -> str:
    """Filter noisy extraction: keep TOC-looking lines to improve parsing.

    Key fixes:
    - Preserve two-column TOC where page numbers are in a separate numeric-only line by
      attaching that numeric line to the previous title line.
    - Keep short title-like lines even if the page number is on the next line.
    """
    if not md:
        return ""
    enable = bool(getattr(cfg.toc, "filter_markdown_lines", True))
    if not enable:
        return md.strip()

    raw_lines = [ln.rstrip() for ln in (md or "").splitlines()]
    kept: List[str] = []

    def _is_numeric_only(s: str) -> bool:
        ss = (s or "").strip()
        if not ss:
            return False
        if re.fullmatch(r"\d{1,5}", ss):
            return True
        if re.fullmatch(r"[ivxlcdm]{1,10}", ss.lower()):
            return True
        return False

    for i, ln in enumerate(raw_lines):
        s = (ln or "").strip()
        if not s:
            continue

        # Merge standalone page-number lines into previous kept title line.
        if _is_numeric_only(s):
            if kept:
                prev = kept[-1]
                if re.search(r"[A-Za-z\u4e00-\u9fff]", prev) and not re.search(r"\d\s*$", prev):
                    kept[-1] = (prev + " " + s).strip()
            continue

        sl = s.lower()

        # Always keep explicit TOC keyword lines.
        if any(k in sl for k in ["contents", "table of contents", "目录", "目 录"]):
            kept.append(s)
            continue

        # Drop very long paragraphs that are unlikely TOC.
        if len(s) > 260 and not re.search(r"\d\s*$", s):
            continue

        # Strong entry signals: trailing page number or dot leader.
        if re.search(r"\d\s*$", s) or re.search(r"\.{3,}\s*\d\s*$", s) or "…" in s:
            kept.append(s)
            continue

        # Short title-like lines: keep if they look like numbered/Chapter entries
        # OR if the next non-empty line is numeric-only (two-column layouts).
        next_nonempty = ""
        for j in range(i + 1, min(i + 4, len(raw_lines))):
            t = (raw_lines[j] or "").strip()
            if t:
                next_nonempty = t
                break

        if re.match(r"^(chapter|part)\b", sl):
            kept.append(s)
            continue
        if re.match(r"^\d+(?:\.\d+)*\b", sl) and re.search(r"[A-Za-z\u4e00-\u9fff]", s):
            kept.append(s)
            continue
        if next_nonempty and _is_numeric_only(next_nonempty) and re.search(r"[A-Za-z\u4e00-\u9fff]", s) and len(s) <= 180:
            kept.append(s)
            continue

    filtered = "\n".join(kept).strip()
    # If filtering removed almost everything, fall back to raw.
    if len(filtered) < 300:
        return (md or "").strip()
    return filtered


def _reduce_to_chapter_level(chapters: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reduce parsed entries to chapter-level only.

    Primary signal: parsed field 'no' (if present). We drop section-like numbers (e.g., 2.3, 1.4.2).
    Fallback signal: title leading numbering.
    """
    in_len = len(chapters or [])
    out: List[Dict[str, Any]] = []
    dropped_section_prefix = 0
    dropped_empty_title = 0

    def is_section_no(no_val: Any, title: str) -> bool:
        s = (str(no_val).strip() if no_val is not None else "")
        if s and re.search(r"\d+\.\d+", s):
            return True
        if re.match(r"^\s*\d+\.\d+", title or ""):
            return True
        return False

    def is_chapter_like(no_val: Any, title: str) -> bool:
        title_s = (title or "").strip()
        tl = title_s.lower()
        if re.search(r"\bappendix\b|附录", tl):
            return True
        s = (str(no_val).strip() if no_val is not None else "")
        if s:
            if re.fullmatch(r"\d{1,3}", s):
                return True
            if re.fullmatch(r"[ivxlcdm]{1,10}", s.lower()):
                return True
        if re.match(r"^(chapter|chap\.?)\s+\d+", tl):
            return True
        if _extract_chapter_no_from_title(title_s) is not None:
            return True
        if re.match(r"^第\s*\d+\s*章", title_s):
            return True
        # "1 Introduction" (not 1.2)
        if re.match(r"^\s*\d{1,3}\s+[A-Za-z\u4e00-\u9fff]", title_s) and not re.match(r"^\s*\d+\.\d+", title_s):
            return True
        return False

    for ch in (chapters or []):
        if not isinstance(ch, dict):
            continue
        title = (ch.get("title") or "").strip()
        if not title:
            dropped_empty_title += 1
            continue
        no_val = ch.get("no")
        if is_section_no(no_val, title) and not is_chapter_like(no_val, title):
            dropped_section_prefix += 1
            continue
        if is_chapter_like(no_val, title):
            out.append(ch)
        else:
            # keep if no is missing but title looks like a chapter heading
            # else drop
            continue

    # Deduplicate by (printed_page, normalized title)
    seen = set()
    dedup: List[Dict[str, Any]] = []
    for ch in out:
        pp = (ch.get("printed_page") or "").strip()
        ttl = re.sub(r"\s+", " ", (ch.get("title") or "").strip().lower())
        key = (pp, ttl)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(ch)

    stats = {
        "in_len": in_len,
        "out_len": len(dedup),
        "dropped_section_prefix": dropped_section_prefix,
        "dropped_empty_title": dropped_empty_title,
    }
    return dedup, stats


def _validate_toc_parse_obj(obj: Dict[str, Any], cfg, *, unit_count: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
    """Validate parsed TOC JSON (chapter-level)."""
    if not isinstance(obj, dict):
        return False, {"reason": "not_a_dict"}
    chapters = obj.get("chapters")
    if not isinstance(chapters, list):
        return False, {"reason": "chapters_not_list"}

    max_chapters = int(getattr(cfg.toc, "max_chapters", 80) or 80)
    if len(chapters) > max_chapters:
        return False, {"reason": "too_many_chapters", "n": len(chapters), "max": max_chapters}

    min_required = _min_required_chapters(cfg, unit_count)
    if len(chapters) < min_required:
        return False, {"reason": "too_few_chapters", "n": len(chapters), "min": min_required}

    # template-title check
    titles = [(ch.get("title") or "").strip() for ch in chapters if isinstance(ch, dict)]
    template_only = sum(1 for t in titles if _is_template_title(t))
    template_ratio = template_only / max(1, len(titles))
    template_max = float(getattr(cfg.toc, "template_title_ratio_max", 0.65) or 0.65)
    if template_ratio > template_max:
        return False, {"reason": "titles_template_only", "ratio": round(template_ratio, 3), "max": template_max}

    # printed_page coverage + span
    pps = [(ch.get("printed_page") or "").strip() for ch in chapters if isinstance(ch, dict)]
    nonempty_pp = sum(1 for pp in pps if pp)
    nonempty_ratio = nonempty_pp / max(1, len(pps))
    min_pp_ratio = float(getattr(cfg.toc, "min_printed_page_nonempty_ratio", 0.60) or 0.60)
    if nonempty_ratio < min_pp_ratio:
        return False, {"reason": "printed_page_too_sparse", "ratio": round(nonempty_ratio, 3), "min": min_pp_ratio}

    # span computed on arabic pages only
    arabic_pages: List[int] = []
    for pp in pps:
        m = re.fullmatch(r"\d{1,4}", pp)
        if m:
            try:
                arabic_pages.append(int(pp))
            except Exception:
                pass
    if len(arabic_pages) >= 3:
        span = max(arabic_pages) - min(arabic_pages)
        min_span = int(getattr(cfg.toc, "min_printed_page_span", 30) or 30)
        if span < min_span:
            return False, {"reason": "printed_page_span_too_small", "span": span, "min": min_span}

    # Starts near chapter 1: use 'no' when possible
    if bool(getattr(cfg.toc, "require_starts_near_one", True)):
        start_max = int(getattr(cfg.toc, "start_near_one_max", 2) or 2)
        nos: List[int] = []
        for ch in chapters:
            if not isinstance(ch, dict):
                continue
            no_val = ch.get("no")
            if isinstance(no_val, int):
                nos.append(no_val)
            else:
                s = (str(no_val).strip() if no_val is not None else "")
                if s.isdigit():
                    nos.append(int(s))
                else:
                    n0 = _cn_numeral_to_int(s)
                    if n0 is not None:
                        nos.append(n0)
                        continue
                    # try title like "Chapter 3"
                    t = (ch.get("title") or "")
                    m = re.match(r"^\s*(?:chapter|chap\.?)\s+(\d{1,3})\b", t, flags=re.IGNORECASE)
                    if m:
                        nos.append(int(m.group(1)))
                    n2 = _extract_chapter_no_from_title(t)
                    if n2 is not None:
                        nos.append(n2)
        if nos:
            if min(nos) > start_max:
                return False, {"reason": "toc_starts_late", "min_no": min(nos), "max_allowed": start_max}

    return True, {"reason": "ok", "n": len(chapters), "template_ratio": round(template_ratio, 3), "pp_nonempty_ratio": round(nonempty_ratio, 3)}


def _repair_template_titles(chapters: List[Dict[str, Any]], md: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Repair template-only chapter titles using nearby lines in markdown.

    Handles patterns like:
        CHAPTER 8
        THE CRYSTALLINE STATE 352
    and Chinese:
        第8章
        晶态 352
    """
    lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
    if not lines or not chapters:
        return chapters, {"repaired": 0}

    # Map chapter number -> candidate title text from markdown
    candidates: Dict[int, str] = {}

    def grab_title_after(idx: int) -> Optional[str]:
        for j in range(idx + 1, min(len(lines), idx + 4)):
            ln = lines[j]
            # ignore pure numbers
            if re.fullmatch(r"\d{1,4}", ln):
                continue
            # ignore another chapter marker
            if re.match(r"^(?:chapter|chap\.?)\s+[0-9ivxlcdm]+", ln, flags=re.IGNORECASE):
                continue
            if re.match(r"^第\s*\d+\s*章", ln):
                continue
            # strip trailing page number
            ln2 = re.sub(r"\s+\d{1,4}\s*$", "", ln).strip()
            if len(ln2) >= 3:
                return ln2
        return None

    for i, ln in enumerate(lines):
        m = re.match(r"^(?:chapter|chap\.?)\s+(\d{1,3})\b", ln, flags=re.IGNORECASE)
        if m:
            n = int(m.group(1))
            # if same line has title text, use it; else look ahead
            rest = re.sub(r"^(?:chapter|chap\.?)\s+\d{1,3}\b", "", ln, flags=re.IGNORECASE).strip(" :.-\t")
            if rest:
                rest = re.sub(r"\s+\d{1,4}\s*$", "", rest).strip()
                if rest:
                    candidates[n] = rest
                    continue
            t2 = grab_title_after(i)
            if t2:
                candidates[n] = t2
            continue

        m2 = re.match(r"^第\s*([0-9０-９]{1,3}|[零〇○一二两三四五六七八九十百千]{1,8})\s*章", ln)
        if m2:
            n = _cn_numeral_to_int(m2.group(1))
            rest = re.sub(r"^第\s*([0-9０-９]{1,3}|[零〇○一二两三四五六七八九十百千]{1,8})\s*章", "", ln).strip(" :.-\t")
            if rest:
                rest = re.sub(r"\s+\d{1,4}\s*$", "", rest).strip()
                if rest:
                    candidates[n] = rest
                    continue
            t2 = grab_title_after(i)
            if t2:
                candidates[n] = t2
            continue

    repaired = 0
    out = []
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        title = (ch.get("title") or "").strip()
        if _is_template_title(title):
            no_val = ch.get("no")
            n = None
            if isinstance(no_val, int):
                n = no_val
            else:
                s = (str(no_val).strip() if no_val is not None else "")
                if s.isdigit():
                    n = int(s)
                else:
                    n0 = _cn_numeral_to_int(s)
                    if n0 is not None:
                        # Chinese numerals in the chapter "no" field (e.g. "一")
                        # should map to an integer chapter number for candidate-title
                        # repair.
                        n = n0
                    else:
                        m = re.match(r"^(?:chapter|chap\.?)\s+(\d{1,3})\b", title, flags=re.IGNORECASE)
                        if m:
                            n = int(m.group(1))
                        m2 = re.match(r"^第\s*([0-9０-９]{1,3}|[零〇○一二两三四五六七八九十百千]{1,8})\s*章", title)
                        if m2:
                            n = _cn_numeral_to_int(m2.group(1))
            if n is not None and n in candidates:
                ch = dict(ch)
                ch["title"] = candidates[n]
                repaired += 1
        out.append(ch)
    return out, {"repaired": repaired, "candidates": len(candidates)}


def _regex_parse_toc_markdown(md: str, cfg) -> Optional[Dict[str, Any]]:
    """Deterministic TOC parse fallback (no LLM).

    This is only used when LLM parsing fails, to avoid a complete pipeline stop.
    It tries to extract chapter-level entries from already-extracted TOC markdown.
    """
    md = (md or "").strip()
    if not md:
        return None

    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    # normalize common bullet / table artifacts
    norm_lines: List[str] = []
    for ln in lines:
        ln2 = re.sub(r"^[\-\*\u2022\s]+", "", ln).strip()
        ln2 = ln2.replace("|", " ")
        ln2 = re.sub(r"\s+", " ", ln2)
        norm_lines.append(ln2)

    def roman_to_int(s: str) -> Optional[int]:
        s = (s or "").strip().lower()
        if not s or not re.fullmatch(r"[ivxlcdm]{1,10}", s):
            return None
        vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
        total = 0
        prev = 0
        for ch in reversed(s):
            v = vals.get(ch, 0)
            if v < prev:
                total -= v
            else:
                total += v
                prev = v
        return total if total > 0 else None

    chapters: List[Dict[str, Any]] = []
    seen: set = set()

    # Patterns (chapter-level only)
    pat_ch_en = re.compile(r"^(?:CHAPTER|Chapter|Chap\.?|CHAP\.?)\s+([0-9]{1,3}|[ivxlcdm]{1,10})\b\s*[:\.\-\s]*([^\d]{1,200}?)\s+(\d{1,4})\s*$")
    pat_num = re.compile(r"^([0-9]{1,3})(?!\.[0-9])\s+(.{1,220}?)\s+(\d{1,4})\s*$")
    pat_ch_cn = re.compile(r"^第\s*([0-9０-９]{1,3}|[零〇○一二两三四五六七八九十百千]{1,8})\s*章\s*[:：\.\-\s]*([^\d]{1,200}?)\s*(\d{1,4})?\s*$")

    for ln in norm_lines[:400]:
        s = ln.strip()
        # Skip obvious section entries like 1.1 / 2.3.4
        if re.match(r"^\d+\.\d+", s):
            continue

        m = pat_ch_en.match(s)
        if m:
            tok = m.group(1)
            title = (m.group(2) or "").strip()
            page = int(m.group(3))
            no = int(tok) if tok.isdigit() else (roman_to_int(tok) or 0)
            if no > 0 and title and (no not in seen):
                chapters.append({"no": no, "title": title, "page": page})
                seen.add(no)
            continue

        m = pat_ch_cn.match(s)
        if m:
            tok = m.group(1)
            title = (m.group(2) or "").strip()
            page_tok = (m.group(3) or "").strip()
            no = None
            if tok.isdigit() or re.fullmatch(r"[0-9０-９]+", tok):
                try:
                    no = int(re.sub(r"[^0-9]", "", tok))
                except Exception:
                    no = None
            else:
                no = _cn_numeral_to_int(tok)
            page = None
            if page_tok:
                try:
                    page = int(page_tok)
                except Exception:
                    page = None
            if isinstance(no, int) and no > 0 and title and (no not in seen):
                rec = {"no": int(no), "title": title}
                if isinstance(page, int):
                    rec["page"] = int(page)
                chapters.append(rec)
                seen.add(int(no))
            continue

        m = pat_num.match(s)
        if m:
            no = int(m.group(1))
            title = (m.group(2) or "").strip()
            page = int(m.group(3))
            # Heuristic: ignore very short titles (often table headers)
            if no > 0 and len(title) >= 4 and (no not in seen):
                chapters.append({"no": no, "title": title, "page": page})
                seen.add(no)
            continue

    if len(chapters) < 2:
        return None

    # sort and truncate
    chapters = sorted(chapters, key=lambda d: int(d.get("no", 0) or 0))
    max_ch = int(getattr(cfg.toc, "max_chapters", 80) or 80)
    chapters = chapters[:max_ch]
    return {"chapters": chapters}


def extract_and_parse_toc(
    store: PDFUnitStore,
    session_text,
    session_vl,
    cfg,
    toc_range: TOCRange,
    cache_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Extract TOC markdown and parse it to structured chapter-level entries.

    Writes:
      - toc_markdown.md
      - toc_attempts.json
      - toc_parse.json
    """
    cache_path = cache_dir / "toc_parse.json"
    cached = load_json(cache_path)
    if cached and isinstance(cached, dict):
        try:
            if int(cached.get("_cache_version", 0)) == int(TOC_CACHE_VERSION):
                return cached
        except Exception:
            pass

    anchors = _get_scan_anchor_units(cache_dir, cfg, toc_range)
    attempts: List[Dict[str, Any]] = []

    max_attempts = int(getattr(cfg.toc, "max_attempts", 3) or 3)
    prefer_text = bool(getattr(cfg.toc, "prefer_text_layer", True))

    # Depth strategy: default strict (chapter_only). Allow extra depths only if configured.
    base_depth = str(getattr(cfg.toc, "parse_depth", "chapter_only") or "chapter_only")
    depth_order = [base_depth]
    if bool(getattr(cfg.toc, "allow_section_fallback", False)):
        for d in ["chapter_section", "entries"]:
            if d not in depth_order:
                depth_order.append(d)

    best_md = ""
    best_md_stats: Dict[str, Any] = {"reason": "none"}

    for attempt in range(1, max_attempts + 1):
        attempt_rec: Dict[str, Any] = {"attempt": attempt, "prefer_text_layer": prefer_text, "anchors": anchors}

        # Extract markdown
        if prefer_text:
            md = _extract_toc_markdown_text_layer(store, toc_range, cfg, anchors=anchors)
            attempt_rec["extract"] = "text_layer"
            if not md and session_vl:
                md = _extract_toc_markdown_vl(store, session_vl, cfg, toc_range, anchors=anchors)
                attempt_rec["extract"] = "vision_vl_fallback"
        else:
            md = _extract_toc_markdown_vl(store, session_vl, cfg, toc_range, anchors=anchors)
            attempt_rec["extract"] = "vision_vl"
            if (not md) and prefer_text:
                md2 = _extract_toc_markdown_text_layer(store, toc_range, cfg, anchors=anchors)
                if md2:
                    md = md2
                    attempt_rec["extract"] = "text_layer_fallback"

        md = filter_toc_markdown((md or "").strip(), cfg)
        ok_md, md_stats = validate_toc_markdown(md, cfg)
        attempt_rec["md_ok"] = bool(ok_md)
        attempt_rec["md_stats"] = md_stats

        # Track best markdown for debugging
        if (ok_md and md_stats.get("md_len", 0) >= best_md_stats.get("md_len", 0)) or (not best_md and md):
            best_md = md
            best_md_stats = md_stats

        if not ok_md:
            attempts.append(attempt_rec)
            # toggle source and (optionally) widen range
            prefer_text = not prefer_text
            if attempt < max_attempts:
                expand = int(getattr(cfg.toc, "range_expand_units", 10) or 10)
                toc_range = TOCRange(
                    max(0, int(toc_range.start_unit) - expand),
                    min(store.unit_count - 1, int(toc_range.end_unit) + expand),
                )
                # recompute anchors within expanded range
                anchors = _get_scan_anchor_units(cache_dir, cfg, toc_range) or anchors
            continue

        # Save markdown once we have a valid candidate
        (cache_dir / "toc_markdown.md").write_text(md, encoding="utf-8")

        # Parse at chosen depth(s)
        enforce_reduce = bool(getattr(cfg.toc, "enforce_chapter_level_reduction", True))
        for depth in depth_order:
            attempt_rec2 = dict(attempt_rec)
            attempt_rec2["parse_depth"] = depth

            try:
                # Compat call: older llm_parse_toc may not accept enable_thinking/temperature.
                call_kwargs = {
                    "model": _resolve_llm_model(cfg),
                    "parse_depth": depth,
                    "max_chapters": int(getattr(cfg.toc, "max_chapters", 80) or 80),
                }

                sig_str = None
                try:
                    sig = inspect.signature(llm_parse_toc)
                    sig_str = str(sig)
                    params = set(sig.parameters.keys())
                    if "enable_thinking" in params:
                        call_kwargs["enable_thinking"] = bool(getattr(cfg.models, "llm_enable_thinking", False))
                    if "temperature" in params:
                        call_kwargs["temperature"] = float(getattr(cfg.models, "llm_temperature", 0.1) or 0.1)
                except Exception:
                    # Best-effort: try passing both, then retry without unexpected kwargs.
                    call_kwargs["enable_thinking"] = bool(getattr(cfg.models, "llm_enable_thinking", False))
                    call_kwargs["temperature"] = float(getattr(cfg.models, "llm_temperature", 0.1) or 0.1)

                try:
                    obj = llm_parse_toc(session_text, md, **call_kwargs)
                except TypeError:
                    # Retry by dropping optional kwargs.
                    call_kwargs.pop("enable_thinking", None)
                    call_kwargs.pop("temperature", None)
                    obj = llm_parse_toc(session_text, md, **call_kwargs)

                attempt_rec2["parse_call"] = {"kwargs": dict(call_kwargs), "signature": sig_str}
            except Exception as e:
                attempt_rec2["parse_ok"] = False
                attempt_rec2["parse_stats"] = {"reason": "llm_parse_error", "error": str(e)}
                attempts.append(attempt_rec2)
                continue
            # Repair template-only titles
            chs = obj.get("chapters") if isinstance(obj, dict) else None
            if isinstance(chs, list):
                repaired_chs, rep_stats = _repair_template_titles(chs, md)
                obj = dict(obj)
                obj["chapters"] = repaired_chs
                attempt_rec2["repair_stats"] = rep_stats

            # Reduce to chapter-level
            reduced, reduce_stats = _reduce_to_chapter_level(obj.get("chapters") or [])
            attempt_rec2["reduce_stats"] = reduce_stats
            min_required = _min_required_chapters(cfg, store.unit_count)
            attempt_rec2["reduce_failed"] = bool(len(reduced) < int(min_required))
            if enforce_reduce:
                if len(reduced) >= int(min_required):
                    obj = dict(obj)
                    obj["chapters"] = reduced
                else:
                    attempt_rec2["parse_ok"] = False
                    attempt_rec2["parse_stats"] = {"reason": "chapter_reducer_insufficient", **reduce_stats}
                    attempts.append(attempt_rec2)
                    continue

            ok_parse, parse_stats = _validate_toc_parse_obj(obj, cfg, unit_count=store.unit_count)
            attempt_rec2["parse_ok"] = bool(ok_parse)
            attempt_rec2["parse_stats"] = parse_stats

            attempts.append(attempt_rec2)
            if ok_parse:
                final_obj = dict(obj)
                final_obj["_cache_version"] = int(TOC_CACHE_VERSION)
                final_obj["_toc_range"] = [int(toc_range.start_unit), int(toc_range.end_unit)]
                final_obj["_anchors"] = anchors
                # Persist attempts and final
                dump_json(cache_dir / "toc_attempts.json", {"attempts": attempts, "best_md_stats": best_md_stats})
                dump_json(cache_path, final_obj)
                return final_obj

        # If parsing failed at all depths, expand and retry
        prefer_text = not prefer_text
        if attempt < max_attempts:
            expand = int(getattr(cfg.toc, "range_expand_units", 10) or 10)
            toc_range = TOCRange(
                max(0, int(toc_range.start_unit) - expand),
                min(store.unit_count - 1, int(toc_range.end_unit) + expand),
            )
            anchors = _get_scan_anchor_units(cache_dir, cfg, toc_range) or anchors

    
    # --- Deterministic regex fallback (when LLM parse fails) ---
    regex_obj = None
    regex_stats: Dict[str, Any] = {}
    try:
        regex_obj = _regex_parse_toc_markdown(best_md, cfg)
        if regex_obj and isinstance(regex_obj, dict):
            ok_obj, obj_stats = _validate_toc_parse_obj(regex_obj, cfg, unit_count=store.unit_count)
            regex_stats = obj_stats or {}
            if ok_obj:
                # persist and return as a non-LLM fallback
                reduced2, reduce_stats2 = _reduce_to_chapter_level((regex_obj or {}).get("chapters") or [])
                if reduced2:
                    regex_obj["chapters"] = reduced2
                    regex_obj["_reduce_stats"] = reduce_stats2
                regex_obj["_fallback"] = "regex_parse"
                regex_obj["_cache_version"] = int(TOC_CACHE_VERSION)
                regex_obj["_toc_range"] = [int(toc_range.start_unit), int(toc_range.end_unit)]
                regex_obj["_anchors"] = anchors
                attempts.append({
                    "attempt": "regex_fallback",
                    "parse_depth": "regex",
                    "md_ok": True,
                    "md_stats": best_md_stats,
                    "parse_ok": True,
                    "parse_stats": obj_stats,
                })
                dump_json(cache_dir / "toc_attempts.json", {"attempts": attempts, "best_md_stats": best_md_stats})
                dump_json(cache_path, regex_obj)
                return regex_obj
            else:
                attempts.append({
                    "attempt": "regex_fallback",
                    "parse_depth": "regex",
                    "md_ok": bool(best_md),
                    "md_stats": best_md_stats,
                    "parse_ok": False,
                    "parse_stats": obj_stats,
                })
    except Exception as e:
        attempts.append({
            "attempt": "regex_fallback",
            "parse_depth": "regex",
            "md_ok": bool(best_md),
            "md_stats": best_md_stats,
            "parse_ok": False,
            "parse_error": str(e),
        })

# All attempts failed: persist diagnostics and return fallback
    dump_json(cache_dir / "toc_attempts.json", {"attempts": attempts, "best_md_stats": best_md_stats})
    fb = {
        "_fallback": True,
        "_fallback_reason": "toc_parse_failed",
        "_cache_version": int(TOC_CACHE_VERSION),
        "_toc_range": [int(toc_range.start_unit), int(toc_range.end_unit)],
        "_anchors": anchors,
        "chapters": [],
    }
    dump_json(cache_path, fb)
    return fb
