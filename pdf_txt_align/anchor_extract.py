from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import re

from .pdf_units import PDFUnitStore
from .llm_calls import vl_extract_opening_anchors
from .utils import dump_json, load_json


ANCHOR_CACHE_VERSION = 2


def _toc_likeness(text: str) -> float:
    """Cheap TOC/Index-likeness score from text layer.

    Used as a guardrail to avoid extracting anchors from TOC/Index pages.
    """
    if not text:
        return 0.0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    tl = text.lower()
    score = 0.0
    if "contents" in tl or "table of contents" in tl or "目录" in tl or "目 录" in tl:
        score += 0.45
    if any(k in tl for k in [
        "index", "references", "bibliography", "glossary", "notation", "symbols", "acknowledg",
        "参考文献", "索引", "附录",
    ]):
        score += 0.25
    tail_digit = sum(1 for ln in lines[:200] if re.search(r"\d\s*$", ln))
    if tail_digit >= 10:
        score += 0.35
    if sum(1 for ln in lines[:200] if re.search(r"\.{3,}\s*\d\s*$", ln) or "…" in ln) >= 8:
        score += 0.20
    return min(1.0, score)


def _symbol_ratio(s: str) -> float:
    """Heuristic: higher ratio indicates formula/TeX/noisy extraction."""
    if not s:
        return 1.0
    t = s.strip()
    if not t:
        return 1.0
    allowed_punct = set(" .,;:!?-'\"()[]{}")
    total = len(t)
    sym = 0
    for ch in t:
        if ch.isalnum() or ch.isspace() or ch in allowed_punct:
            continue
        # allow basic CJK letters
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:
            continue
        sym += 1
    return sym / max(1, total)


def _normalize_anchors(anchors: Any, *, max_len: int) -> List[str]:
    if isinstance(anchors, str):
        anchors = [anchors]
    if not isinstance(anchors, list):
        return []
    out: List[str] = []
    for a in anchors:
        s = str(a or "").strip()
        if not s:
            continue
        if len(s) > max_len:
            s = s[:max_len].rstrip()
        out.append(s)
    # de-dup
    uniq: List[str] = []
    seen = set()
    for a in out:
        key = a.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    return uniq




def _derive_pdf_heading_hint(title: str, opening_snippet: str, anchors: List[str]) -> str:
    """Extract a chapter-heading-like hint from VL outputs for downstream verification/debugging."""
    cands: List[str] = []
    for x in [title] + list(anchors or []):
        t = str(x or "").strip()
        if t:
            cands.append(t)
    for ln in str(opening_snippet or "").splitlines()[:8]:
        t = ln.strip()
        if t:
            cands.append(t)
    for t in cands:
        if re.search(r"(?i)\bchapter\b", t) or re.search(r"第\s*[0-9０-９零〇○一二三四五六七八九十百千]+\s*章", t):
            return t[:240]
    return (str(title or "").strip()[:240])

def extract_anchors_for_chapters(
    store: PDFUnitStore,
    session_vl,
    cfg,
    chapters: List[Dict[str, Any]],
    cache_dir: Path,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Extract robust textual anchors from around each chapter start unit.

    Extends each chapter dict with:
      - anchors: [string]
      - opening_snippet: string
      - anchor_conf: float
      - anchor_unit: int|None
      - anchor_tried_units: [int]
      - anchor_symbol_ratio: float

    NOTE:
    - Default search is forward-only from unit_start to reduce picking TOC/Index/Author pages.
    - Backward search is used ONLY as a fallback if forward search yields no usable candidate.
    """

    out_path = cache_dir / "chapter_anchors.json"
    cached = load_json(out_path)
    if cached and isinstance(cached, list) and len(cached) == len(chapters):
        try:
            ok_ver = all(isinstance(x, dict) and int(x.get("_anchor_cache_version", 0) or 0) == ANCHOR_CACHE_VERSION for x in cached)
            if ok_ver:
                return cached
        except Exception:
            pass

    radius = int(getattr(cfg.anchors, "unit_search_radius", 2) or 2)
    reject_score = float(getattr(cfg.anchors, "reject_toc_score", 0.70) or 0.70)
    sym_max = float(getattr(cfg.anchors, "symbol_ratio_max", 0.55) or 0.55)
    max_len = int(getattr(cfg.anchors, "anchor_max_len_chars", 180) or 180)

    forward_only = bool(getattr(cfg.anchors, "search_forward_only", True))
    allow_backward = bool(getattr(cfg.anchors, "allow_backward_fallback", True))

    # Title overlap guard (prevents picking "About the author" etc.)
    min_overlap = float(getattr(cfg.anchors, "min_title_token_overlap", 0.15) or 0.15)

    vision_model = str(getattr(cfg.models, "vision_model", "qwen3-vl-plus"))
    enable_thinking = bool(getattr(cfg.models, "vision_enable_thinking", False))
    snippet_words = int(getattr(cfg.anchors, "opening_snippet_words", 220) or 220)
    anchors_per = int(getattr(cfg.anchors, "anchors_per_chapter", 3) or 3)

    stop = {
        "chapter", "part", "section", "contents", "table", "of", "the", "and",
        "目录", "目", "录",
    }
    bad_kw = [
        "about the author", "about the authors", "author", "authors",
        "preface", "acknowledg", "acknowledgement", "acknowledgments",
        "references", "bibliography", "index", "glossary", "notation", "symbols",
        "contents", "table of contents", "目录", "目 录", "索引", "参考文献",
    ]

    def _tokens(s: str) -> List[str]:
        if not s:
            return []
        # split into alnum/CJK groups
        toks = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", s.lower())
        out = []
        for t in toks:
            if t in stop:
                continue
            if len(t) <= 1:
                continue
            out.append(t)
        return out

    def _overlap(a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        inter = len(sa & sb)
        denom = max(1, min(len(sa), len(sb)))
        return inter / denom

    def _chapter_no_match(s: str, no: int) -> bool:
        if not s or not no:
            return False
        sl = s.lower()
        if re.search(rf"\bchapter\s+{no}\b", sl):
            return True
        if re.search(rf"\bch\.?\s*{no}\b", sl):
            return True
        return False

    def _looks_bad_heading(s: str) -> bool:
        if not s:
            return False
        sl = s.lower()
        return any(k in sl for k in bad_kw)

    results: List[Dict[str, Any]] = []

    for ch in chapters:
        unit_start = ch.get("unit_start", None)
        if unit_start is None:
            results.append({
                **ch,
                "anchors": [],
                "opening_snippet": "",
                "pdf_heading_hint": "",
                "anchor_conf": 0.0,
                "anchor_symbol_ratio": 1.0,
                "anchor_unit": None,
                "anchor_tried_units": [],
                "_anchor_cache_version": ANCHOR_CACHE_VERSION,
            })
            continue

        no = int(ch.get("no", 0) or 0)
        us = int(unit_start)
        lo = max(0, us - radius)
        hi = min(store.unit_count - 1, us + radius)

        # search order: forward first; backward only if forward yields nothing usable.
        forward_units = list(range(us, hi + 1))
        backward_units = list(range(us - 1, lo - 1, -1))

        tried: List[int] = []

        toc_title = str(ch.get("title_corrected") or ch.get("title") or "").strip()
        toc_tokens = _tokens(toc_title)

        best_ok_good: Optional[Tuple[float, int, Dict[str, Any], float]] = None  # (conf, unit, info, sym_ratio)
        best_ok_any: Optional[Tuple[float, int, Dict[str, Any], float]] = None
        best_any: Optional[Tuple[float, int, Dict[str, Any], float]] = None

        def _try_units(units: List[int]) -> None:
            nonlocal best_ok_good, best_ok_any, best_any
            for ui in units:
                tried.append(int(ui))

                # Guardrail: skip TOC/Index-like pages by text layer.
                try:
                    page_text = store.extract_unit_text(store.unit_ref(int(ui)), region="full") or ""
                    if page_text and _toc_likeness(page_text) >= reject_score:
                        continue
                except Exception:
                    pass

                img = store.render_unit(store.unit_ref(int(ui)), dpi=store.dpi_high, region="full")
                info = vl_extract_opening_anchors(
                    session_vl,
                    img,
                    no,
                    model=vision_model,
                    enable_thinking=enable_thinking,
                    snippet_words=snippet_words,
                    anchors_per_chapter=anchors_per,
                ) or {}

                conf = float(info.get("conf", 0.0) or 0.0)
                anchors = _normalize_anchors(info.get("anchors"), max_len=max_len)
                opening_snippet = str(info.get("opening_snippet", "") or "").strip()
                title_vl = str(info.get("title", "") or "").strip()

                if not anchors and not title_vl:
                    continue

                combined = "\n".join(anchors[:3] + ([opening_snippet] if opening_snippet else []) + ([title_vl] if title_vl else []))
                sym = _symbol_ratio(combined)

                cand = (conf, int(ui), info, sym)
                if best_any is None or cand[0] > best_any[0]:
                    best_any = cand

                if sym <= sym_max:
                    if best_ok_any is None or cand[0] > best_ok_any[0]:
                        best_ok_any = cand

                    # Additional semantic guard: avoid "bad heading" unless it overlaps TOC title or matches chapter no.
                    cand_tokens = _tokens(" ".join([title_vl] + anchors[:2] + ([opening_snippet] if opening_snippet else [])))
                    ov = _overlap(toc_tokens, cand_tokens)
                    bad = _looks_bad_heading(title_vl + " " + opening_snippet)
                    chap_ok = _chapter_no_match(title_vl + " " + " ".join(anchors[:2]), no)

                    good = True
                    if toc_tokens and len(toc_tokens) >= 2:
                        if (ov < min_overlap) and (not chap_ok) and bad:
                            good = False
                    # If toc title is generic (few tokens), do not over-filter.
                    if good:
                        if best_ok_good is None or cand[0] > best_ok_good[0]:
                            best_ok_good = cand

        _try_units(forward_units)

        # Backward fallback:
        # - If forward-only: try the immediate previous unit(s) only when we did not find a "good" candidate.
        # - If not forward-only: allow full backward radius.
        if best_ok_good is None and allow_backward and backward_units:
            if forward_only:
                # If forward yielded only weak/bad headings, try one step backward (often the actual chapter opening).
                _try_units(backward_units[: min(2, len(backward_units))])
                # If forward yielded nothing at all, expand backward search.
                if best_ok_good is None and best_ok_any is None and best_any is None:
                    _try_units(backward_units)
            else:
                _try_units(backward_units)

        chosen = best_ok_good or best_ok_any or best_any
        if chosen is None:
            results.append({
                **ch,
                "anchors": [],
                "opening_snippet": "",
                "pdf_heading_hint": "",
                "anchor_conf": 0.0,
                "anchor_symbol_ratio": 1.0,
                "anchor_unit": int(unit_start),
                "anchor_tried_units": tried,
                "_anchor_cache_version": ANCHOR_CACHE_VERSION,
            })
            continue

        conf, best_unit, info, sym = chosen
        anchors = _normalize_anchors(info.get("anchors"), max_len=max_len)
        opening_snippet = str(info.get("opening_snippet", "") or "").strip()
        title_vl = str(info.get("title", "") or "").strip()

        enriched = {
            **ch,
            "anchors": anchors,
            "opening_snippet": opening_snippet,
            "title_vl": (str(title_vl).strip() if isinstance(title_vl, str) and str(title_vl).strip() else None),
            "pdf_heading_hint": _derive_pdf_heading_hint(title_vl or ch.get("title_corrected") or ch.get("title"), opening_snippet, anchors),
            "anchor_conf": float(conf),
            "anchor_symbol_ratio": float(sym),
            "anchor_unit": int(best_unit),
            "anchor_tried_units": tried,
            "_anchor_cache_version": ANCHOR_CACHE_VERSION,
        }

        # Only accept VL title correction if it is consistent with TOC title or chapter no.
        if title_vl:
            toc_tokens = _tokens(str(ch.get("title") or ""))
            cand_tokens = _tokens(title_vl)
            ov = _overlap(toc_tokens, cand_tokens)
            chap_ok = _chapter_no_match(title_vl, no)
            if chap_ok or (toc_tokens and ov >= 0.20):
                # Also avoid overriding with clearly non-body headings.
                if not (_looks_bad_heading(title_vl) and (not chap_ok) and ov < min_overlap):
                    enriched["title_corrected"] = title_vl

        results.append(enriched)

    dump_json(out_path, results)
    return results
