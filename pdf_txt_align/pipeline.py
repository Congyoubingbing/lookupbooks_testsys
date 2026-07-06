from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import traceback
import re
import os
import shutil
import hashlib
from datetime import datetime

from .config import load_config
from .utils import ensure_dir, dump_json, load_json, sanitize_filename, safe_output_filename, now_ts
from .pdf_units import PDFUnitStore
from .toc import find_toc_range, extract_and_parse_toc
from .page_locator import PageLocator
from .anchor_extract import extract_anchors_for_chapters
from .text_align import boundaries_from_tex_headings, boundaries_by_anchor_search
from .verify import verify_chapter_segments
from .text_processing import (
    parse_tex_chapters,
    parse_plain_chapters,
    parse_tex_headings,
    build_tex_shadow_same_len,
    normalize_text_for_match,
)
from .llm_calls import call_json, vl_verify_chapter_start
from .sequence_align import Chapter as Ch
from .similarity import ratio, partial_ratio, tokenize_simple, title_similarity, normalize_title_robust
from .utils import normalize_title
from .safe_cut import snap_boundaries

def _persist_and_cleanup_cache(book_out: Path, cache_dir: Path, cfg, summary: Dict[str, Any], logger) -> None:
    """Persist lightweight, reusable artifacts from cache_dir into book_out/artifacts,
    then remove cache_dir if runtime.keep_intermediate* is disabled.

    This satisfies lookupbooks_sys requirement: delete large intermediate products
    (especially cached images), while keeping final split results and compact
    structure files usable by downstream agents.
    """
    # Feature flag (default: delete intermediates)
    keep_cache = bool(
        getattr(getattr(cfg, "runtime", None), "keep_intermediate", False)
        or getattr(getattr(cfg, "runtime", None), "keep_cache", False)
        or getattr(getattr(cfg, "runtime", None), "keep_intermediate_cache", False)
    )
    persist = bool(getattr(getattr(cfg, "runtime", None), "persist_cache_artifacts", True))
    if keep_cache:
        return
    if not cache_dir.exists():
        return

    artifacts_dir = ensure_dir(book_out / "artifacts")

    # Only keep compact, reusable artifacts. DO NOT keep cached page/unit images.
    keep_files = [
        ("meta.json", "meta.json"),
        ("toc_scan.json", "toc_scan.json"),
        ("toc_markdown.md", "toc_markdown.md"),
        ("toc_attempts.json", "toc_attempts.json"),
        ("toc_parse.json", "toc_parse.json"),
        ("chapter_plan.json", "chapter_plan.json"),
        ("chapters_with_anchors.json", "chapters_with_anchors.json"),
        ("chapter_anchors.json", "chapter_anchors.json"),
        ("chapters_mapped.json", "chapters_mapped.json"),
        ("chapter_unit_monotone_repair.json", "chapter_unit_monotone_repair.json"),
        ("boundary_decisions.json", "boundary_decisions.json"),
        ("text_boundaries.json", "text_boundaries.json"),
        ("text_boundaries_final.json", "text_boundaries_final.json"),
        ("verify_report.json", "verify_report.json"),
        ("cut_snap_report.json", "cut_snap_report.json"),
    ]

    if persist:
        for src_name, dst_name in keep_files:
            src = cache_dir / src_name
            if not src.exists():
                continue
            try:
                dst = artifacts_dir / dst_name
                shutil.copy2(src, dst)
            except Exception:
                pass

        # Rewrite summary artifact pointers if they referenced cache paths
        try:
            # normalize existing artifact paths to artifacts/ when applicable
            for k, v in list((summary.get("artifacts") or {}).items()):
                if isinstance(v, str) and str(cache_dir) in v:
                    # if file was persisted, point to new path
                    base = Path(v).name
                    cand = artifacts_dir / base
                    if cand.exists():
                        summary["artifacts"][k] = str(cand)
        except Exception:
            pass
        try:
            summary["artifacts"]["artifacts_dir"] = str(artifacts_dir)
        except Exception:
            pass

    # Remove cache dir (including images/) to satisfy "delete intermediates"
    try:
        shutil.rmtree(cache_dir, ignore_errors=True)
        logger.info("[CLEAN] removed cache_dir=%s", str(cache_dir))
    except Exception:
        pass


def _norm_for_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _local_snippet_score(opening_snippet: str, window_text: str) -> float:
    a = _norm_for_match(opening_snippet)
    if not a:
        return 0.0
    b = _norm_for_match(window_text)
    if not b:
        return 0.0
    return float(partial_ratio(a[:400], b[:5000]) or 0.0)


def _local_snippet_alignment(opening_snippet: str, window_text: str, *, scan_chars: int = 20000) -> Tuple[float, Optional[int], Optional[float]]:
    """Compute local snippet score plus an approximate lead offset of the best match.

    partial_ratio can be high even if the best match occurs far from the boundary. We use
    rapidfuzz.partial_ratio_alignment to estimate where the match starts (lead offset).

    Returns:
      (score in [0,1], lead_offset in chars, lead_score in [0,1])
    """
    score = _local_snippet_score(opening_snippet, window_text)
    if not opening_snippet or not window_text:
        return score, None, None
    try:
        from rapidfuzz.fuzz import partial_ratio_alignment
    except Exception:
        return score, None, None

    a = (opening_snippet or "").strip()
    if len(a) > 1800:
        a = a[:1800]
    b = (window_text or "")[: max(0, int(scan_chars or 0))]
    if not a or not b:
        return score, None, None
    try:
        al = partial_ratio_alignment(a.lower(), b.lower())
        if not al:
            return score, None, None
        lead = int(getattr(al, "dest_start", -1) or -1)
        lead_score = float(getattr(al, "score", 0.0) or 0.0) / 100.0
        if lead < 0:
            lead = None
        return score, lead, lead_score
    except Exception:
        return score, None, None

def _pick_initial_boundary_candidate(candidates_init: List[Tuple[str, int, float]], cfg) -> Tuple[Optional[str], Optional[int], float, List[str]]:
    """Pick the initial boundary candidate with TeX-heading preference policy."""
    if not candidates_init:
        return None, None, 0.0, []

    cands: List[Tuple[str, int, float]] = []
    for m, pos, conf in candidates_init:
        try:
            p = int(pos)
        except Exception:
            continue
        if p < 0:
            continue
        cands.append((str(m), p, float(conf or 0.0)))
    if not cands:
        return None, None, 0.0, []
    cands.sort(key=lambda t: t[2], reverse=True)

    flags: List[str] = []
    if bool(getattr(cfg.align, "prefer_tex_headings", True)):
        tex_pref_min_conf = float(getattr(cfg.align, "tex_heading_pref_min_conf", 0.05) or 0.05)

        def _tex_rank(method: str) -> int:
            m = str(method or "")
            if not m.startswith("tex_"):
                return 99
            if m.startswith("tex_chapter"):
                return 0
            if m.startswith("tex_part"):
                return 1
            if m.startswith("tex_section"):
                return 2
            if m.startswith("tex_subsection"):
                return 3
            return 4

        tex_cands = [c for c in cands if c[0].startswith("tex_") and float(c[2]) >= tex_pref_min_conf]
        if tex_cands:
            tex_cands.sort(key=lambda t: (_tex_rank(t[0]), -float(t[2]), int(t[1])))
            picked = tex_cands[0]
            if picked != cands[0]:
                flags.append("prefer_tex_selected")
            return picked[0], picked[1], picked[2], flags

    picked = cands[0]
    return picked[0], picked[1], picked[2], flags

    a = (opening_snippet or "").strip()
    if len(a) > 1800:
        a = a[:1800]
    b = (window_text or "")[: max(0, int(scan_chars or 0))]
    if not a or not b:
        return score, None, None
    try:
        al = partial_ratio_alignment(a.lower(), b.lower())
        if not al:
            return score, None, None
        lead = int(getattr(al, "dest_start", -1) or -1)
        lead_score = float(getattr(al, "score", 0.0) or 0.0) / 100.0
        if lead < 0:
            lead = None
        return score, lead, lead_score
    except Exception:
        return score, None, None


def _toc_likeness(window_text: str) -> float:
    """Heuristic score in [0..1] indicating the window resembles a TOC/Index."""
    t = (window_text or "")
    if not t:
        return 0.0
    # downsample
    t = t[:20000].lower()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    n = len(lines)
    # lines ending with numbers are TOC-like
    end_num = 0
    dot_leader = 0
    for ln in lines[:400]:
        if re.search(r"(\.|·|\s)\s*\d{1,4}\s*$", ln):
            end_num += 1
        flag_dot = False
        if "...." in ln or "··" in ln:
            flag_dot = True
        # TeX dot leaders: \\dotfill / \\leaders\hbox{.}\\hfill
        if ("dotfill" in ln) or ("leaders" in ln and ("hfill" in ln or "hbox" in ln or "to" in ln)):
            flag_dot = True
        if flag_dot:
            dot_leader += 1
    r_end = end_num / float(max(1, min(400, n)))
    r_dot = dot_leader / float(max(1, min(400, n)))
    kw = 1.0 if ("contents" in t or "table of contents" in t or "index" in t) else 0.0
    # combine
    score = 0.55 * r_end + 0.25 * r_dot + 0.2 * kw
    return float(max(0.0, min(1.0, score)))



def _is_section_like_integer_dot_title(title: str) -> bool:
    t = str(title or "").strip()
    if not t:
        return False
    if re.match(r"^\s*[0-9]{1,3}\s*[.．。]\s+", t):
        return True
    if re.match(r"^\s*[0-9]{1,3}(?:\s*[.．。]\s*[0-9]{1,3})+", t):
        return True
    return False


def _looks_like_hashy_title(title: str) -> bool:
    t = str(title or "").strip()
    return bool(t) and bool(re.fullmatch(r"[0-9a-f]{8,32}", t.lower()))


def _extract_segment_start_tex_chapter_title(raw_text: str, start: int, *, window: int = 2400) -> str:
    """Extract a chapter-level title near segment start for filename correction."""
    try:
        s0 = max(0, int(start))
        sub = raw_text[s0:min(len(raw_text), s0 + int(window))]
        if not sub:
            return ""
        evs = parse_tex_chapters(
            sub,
            toc_titles=None,
            min_chapters=1,
            dedup_gap_chars=200,
            toc_title_promotion_min_sim=0.95,
        ) or []
        if evs:
            ev0 = evs[0]
            if int(getattr(ev0, 'start', 10**9)) <= 120:
                return str(getattr(ev0, 'title', '') or '').strip()
    except Exception:
        return ""
    return ""


def _resolve_output_chapter_title(ch: Dict[str, Any], cfg=None) -> str:
    """Resolve output filename title, preferring chapter-level signals over section-like TOC noise."""
    toc_title = str(ch.get("title_corrected") or ch.get("title") or "").strip()
    seg_title = str(ch.get("segment_start_title") or "").strip()
    tex_title = str(ch.get("matched_tex_heading") or "").strip()
    pdf_title = str(ch.get("matched_pdf_heading") or ch.get("title_vl") or "").strip()

    def _good(t: str) -> bool:
        return bool(t) and (not _is_section_like_integer_dot_title(t)) and (not _looks_like_hashy_title(t))

    # Strongest signal: explicit chapter title found right at the emitted slice start.
    if _good(seg_title):
        return seg_title

    cands = [("tex", tex_title), ("pdf", pdf_title), ("toc", toc_title)]
    cands = [(src, t) for src, t in cands if t]
    if not cands:
        return toc_title

    toc_sus = _is_section_like_integer_dot_title(toc_title) or _looks_like_hashy_title(toc_title)
    if toc_sus:
        for src, t in cands:
            if src in {"tex", "pdf"} and _good(t):
                return t

    for src, t in cands:
        if src in {"tex", "pdf"} and _good(t):
            return t

    return cands[0][1]


def _trim_backmatter_from_last_chapter(raw_text: str, last_start: int, cfg) -> Tuple[int, Optional[str]]:
    if not raw_text:
        return 0, None
    try:
        enable = bool(getattr(cfg.align, "backmatter_trim_enable", True))
    except Exception:
        enable = True
    if not enable:
        return len(raw_text), None
    try:
        kws = list(getattr(cfg.align, "backmatter_trim_keywords", []) or [])
    except Exception:
        kws = []
    if not kws:
        kws = [
            r"\section*{APPENDIX}", r"\chapter{APPENDIX}",
            r"\section*{BIBLIOGRAPHY}", r"\section*{Bibliography}",
            r"\section*{REFERENCES}", r"\section*{References}",
            r"\section*{SUBJECT INDEX}", r"\section*{AUTHOR INDEX}",
            r"\section*{参考文献}", r"\section*{索引}",
        ]
    seg = raw_text[max(0, int(last_start)):]
    best_pos = None
    best_kw = None
    for kw in kws:
        kws_ = str(kw or "")
        if not kws_:
            continue
        pos = seg.find(kws_)
        if pos < 0:
            plain = re.sub(r"\\[A-Za-z]+\*?\{", "", kws_)
            plain = plain.replace("}", "")
            plain = re.sub(r"\s+", " ", plain).strip()
            if plain:
                m2 = re.search(re.escape(plain), seg, flags=re.IGNORECASE)
                pos = (m2.start() if m2 else -1)
        if pos >= 0 and (best_pos is None or pos < best_pos):
            best_pos = pos
            best_kw = kws_
    if best_pos is None:
        return len(raw_text), None
    return max(0, int(last_start)) + int(best_pos), best_kw


def _pdf_text_layer_heading_probe(store: PDFUnitStore, chapter_title: str, unit_idx: int, cfg, *, radius_units: int = 3) -> Dict[str, Any]:
    out: Dict[str, Any] = {"hit": False, "unit": int(unit_idx), "score": 0.0, "matched": None}
    if store is None or not chapter_title:
        return out
    try:
        doc = getattr(store, "_doc", None) or getattr(store, "doc", None)
        unit_count = int(getattr(store, "unit_count", 0) or 0)
    except Exception:
        return out
    if doc is None or unit_count <= 0:
        return out
    q = normalize_title_robust(chapter_title or "")
    if not q:
        return out
    lo = max(0, int(unit_idx) - max(0, int(radius_units)))
    hi = min(unit_count - 1, int(unit_idx) + max(0, int(radius_units)))
    best = None
    for u in range(lo, hi + 1):
        try:
            ref = store.unit_ref(int(u))
            page_idx = int(ref[0])
        except Exception:
            page_idx = -1
        if page_idx < 0:
            continue
        try:
            page = doc.load_page(page_idx)
            txt = page.get_text("text") or ""
        except Exception:
            txt = ""
        if not txt:
            continue
        txt_norm = normalize_title_robust(txt)
        if not txt_norm:
            continue
        sim = float(title_similarity(q, txt_norm) or 0.0)
        toks = [t for t in re.split(r"\s+", q) if len(t) >= 3][:8]
        if toks:
            hits = sum(1 for t in toks if t in txt_norm)
            sim = min(1.0, sim + 0.15 * (hits / max(1, len(toks))))
        sample = txt.strip().splitlines()[0][:200] if txt.strip() else ""
        cand = (sim, u, sample)
        if best is None or cand[0] > best[0]:
            best = cand
    min_sim = float(getattr(cfg.align, "pdf_text_probe_min_sim", 0.75) or 0.75)
    if best:
        out.update({"unit": int(best[1]), "score": round(float(best[0]), 4), "matched": str(best[2])})
        out["hit"] = bool(float(best[0]) >= min_sim)
    return out



_CN_NUM = {"零":0,"〇":0,"○":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
_CN_UNIT = {"十":10,"百":100,"千":1000}


def _cn_int(s: str) -> Optional[int]:
    t = (s or "").strip()
    if not t:
        return None
    fw = str.maketrans({"０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9"})
    t = t.translate(fw)
    if t.isdigit():
        try:
            return int(t)
        except Exception:
            return None
    total = 0
    num = 0
    seen_unit = False
    for ch in t:
        if ch in _CN_NUM:
            num = _CN_NUM[ch]
        elif ch in _CN_UNIT:
            seen_unit = True
            u = _CN_UNIT[ch]
            if num == 0:
                num = 1
            total += num * u
            num = 0
    total += num
    if total == 0 and seen_unit:
        total = 10
    return int(total) if total > 0 else None


def _extract_chapter_no_from_text(title: str) -> Optional[int]:
    t = (title or "").strip()
    if not t:
        return None
    m = re.search(r"(?i)\b(?:chapter|chap\.?)\s*([0-9]{1,3})\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    m = re.search(r"第\s*([0-9０-９零〇○一二两三四五六七八九十百千]{1,8})\s*章", t)
    if m:
        return _cn_int(m.group(1))
    # Reject decimal section prefixes like "7.1" / "3.2".
    if re.match(r"^\s*[0-9]{1,3}\s*[.．。]\s*[0-9]{1,3}\b", t):
        return None
    # Allow integer-only chapter-like headings, e.g. "7 MOLECULAR THEORY ..."
    m = re.match(r"^\s*([0-9]{1,3})(?!\s*[.．。]\s*[0-9])\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None


def _is_chapterlike_heading(kind: str, title: str) -> bool:
    k = (kind or "").lower()
    t = (title or "").strip()
    if not t:
        return False
    if k in {"chapter", "part"}:
        return True
    # Many TeX sources encode chapter boundaries as \section*{Chapter 4} / \section*{第七章 ...}
    return _extract_chapter_no_from_text(t) is not None


def _is_decimal_section_heading_title(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return False
    return bool(re.match(r"^\s*[0-9]{1,3}\s*[.．。]\s*[0-9]{1,3}\b", t))


def _is_strict_chapterlike_heading(kind: str, title: str) -> bool:
    k = (kind or "").lower()
    t = (title or "").strip()
    if not t:
        return False
    if k in {"chapter", "part"}:
        return True
    if re.search(r"(?i)\b(?:chapter|chap\.?)\s*[0-9]{1,3}\b", t):
        return True
    if re.search(r"第\s*([0-9０-９零〇○一二两三四五六七八九十百千]{1,8})\s*章", t):
        return True
    return (_extract_chapter_no_from_text(t) is not None) and (not _is_decimal_section_heading_title(t))


def _title_sim_loose(a: str, b: str) -> float:
    na = normalize_title(a or "")
    nb = normalize_title(b or "")
    if not na or not nb:
        return 0.0
    try:
        return float(ratio(na, nb) or 0.0)
    except Exception:
        return 0.0


def _find_tex_heading_near(
    tex_heading_events: List[Any],
    *,
    chapter_no: Optional[int],
    chapter_title: str,
    around: int,
    window: int,
    prefer_side: str = "either",  # 'before' | 'after' | 'either'
    chapter_title_fallback: str = "",
    promote_min_sim: float = 0.72,
) -> Optional[Dict[str, Any]]:
    """Find a chapter-level TeX heading near a proposed boundary.

    Strict pass avoids the historical false positive where decimal section headings
    (e.g., 7.1 / 3.2) hijack the chapter boundary because they share the same leading number.
    """
    if not tex_heading_events:
        return None
    lo = int(around) - int(max(0, window))
    hi = int(around) + int(max(0, window))
    no_target = int(chapter_no) if isinstance(chapter_no, int) and chapter_no > 0 else None
    side = str(prefer_side or "either").lower()
    title_primary = str(chapter_title or "")
    title_fallback = str(chapter_title_fallback or "")

    def _scan(strict_only: bool, stage: str) -> Optional[Dict[str, Any]]:
        best = None
        best_score = -10**9
        for e in tex_heading_events:
            try:
                pos = int(getattr(e, "start", -1) or -1)
            except Exception:
                continue
            if pos < lo or pos > hi:
                continue
            delta = pos - int(around)
            if side == "before" and delta > 0:
                continue
            if side == "after" and delta < 0:
                continue
            kind = str(getattr(e, "kind", "") or "").lower()
            title = str(getattr(e, "title", "") or "")
            if strict_only:
                if not _is_strict_chapterlike_heading(kind, title):
                    continue
            else:
                is_chapterlike = _is_chapterlike_heading(kind, title)
                if not is_chapterlike:
                    title_sim_probe = max(
                        _title_sim_loose(title_primary, title) if title_primary else 0.0,
                        _title_sim_loose(title_fallback, title) if title_fallback else 0.0,
                    )
                    looks_int_dot = bool(re.match(r"^\d{1,3}\s*\.\s+", (title or "").strip()))
                    if not (
                        kind in {"chapter", "part", "section", "subsection"}
                        and (not looks_int_dot)
                        and float(title_sim_probe) >= float(promote_min_sim)
                    ):
                        continue

            eno = getattr(e, "no", None)
            eno_i = None
            try:
                if eno is not None:
                    eno_i = int(eno)
            except Exception:
                eno_i = None
            inf_no = _extract_chapter_no_from_text(title)

            no_match = (no_target is None)
            if no_target is not None:
                if eno_i is not None and eno_i == no_target:
                    no_match = True
                elif inf_no is not None and inf_no == no_target:
                    no_match = True
            title_sim_for_no = max(
                _title_sim_loose(title_primary, title) if title_primary else 0.0,
                _title_sim_loose(title_fallback, title) if title_fallback else 0.0,
            )
            if not no_match and float(title_sim_for_no) < max(0.90, float(promote_min_sim) + 0.12):
                continue

            title_sim = max(
                _title_sim_loose(title_primary, title) if title_primary else 0.0,
                _title_sim_loose(title_fallback, title) if title_fallback else 0.0,
            )
            is_decimal = _is_decimal_section_heading_title(title)
            if _is_section_like_integer_dot_title(title):
                continue
            is_strict = _is_strict_chapterlike_heading(kind, title)

            score = 0.0
            score += 8.0
            if kind in {"chapter", "part"}:
                score += 5.0
            if is_strict:
                score += 3.0
            if is_decimal:
                score -= 6.0
            score += 4.0 * float(title_sim)
            score -= abs(delta) / max(1.0, float(max(100, window)))
            if side == "before" and delta <= 0:
                score += 0.5
            if side == "after" and delta >= 0:
                score += 0.5

            if score > best_score:
                best_score = score
                best = {
                    "pos": int(pos),
                    "delta": int(delta),
                    "kind": kind,
                    "title": title,
                    "event_no": eno_i,
                    "inferred_no": inf_no,
                    "title_sim": round(float(title_sim), 4),
                    "score": round(float(score), 4),
                    "strict": bool(is_strict),
                    "decimal": bool(is_decimal),
                    "stage": stage,
                }
        return best

    hit = _scan(True, "strict")
    if hit is None:
        hit = _scan(False, "loose")
    return hit


def _apply_tex_heading_lock(
    bounds: List[int],
    boundary_details: List[Dict[str, Any]],
    chapters_anchored: List[Dict[str, Any]],
    tex_heading_events: List[Any],
    *,
    cfg,
) -> Tuple[List[int], Dict[str, Any]]:
    """Force boundaries onto nearby TeX chapter headings, with de-dup and spacing guards."""
    if not bounds or not tex_heading_events:
        return list(bounds or []), {'enabled': False, 'items': []}
    enable = bool(getattr(cfg.align, 'tex_heading_lock_enable', True))
    if not enable:
        return list(bounds), {'enabled': False, 'items': []}

    win = int(getattr(cfg.align, 'tex_heading_lock_window_chars', 20000) or 20000)
    force_after = int(getattr(cfg.align, 'tex_heading_lock_force_after_heading_chars', 120000) or 120000)
    force_before = int(getattr(cfg.align, 'tex_heading_lock_force_before_heading_chars', 40000) or 40000)
    min_title_sim = float(getattr(cfg.align, 'tex_heading_lock_min_title_sim', 0.45) or 0.45)
    promote_min_sim = float(getattr(cfg.align, 'tex_toc_title_promotion_min_sim', 0.72) or 0.72)
    dual_enable = bool(getattr(cfg.align, 'tex_heading_lock_dual_boundary_enable', True))
    reclaim_enable = bool(getattr(cfg.align, 'tex_heading_lock_reclaim_next_heading', True))
    mg = int(getattr(cfg.align, 'tex_heading_lock_min_gap_chars', 32) or 32)
    min_prev_seg = int(getattr(cfg.align, 'tex_heading_lock_min_prev_seg_chars', 64) or 64)
    cluster_sep = int(getattr(cfg.align, 'tex_heading_lock_min_cluster_sep_chars', 8) or 8)

    out = [int(x) for x in bounds]
    items: List[Dict[str, Any]] = []
    used_heading_pos: List[int] = []

    for i, ch in enumerate(chapters_anchored or []):
        if i >= len(out):
            break
        cur = int(out[i])
        no = int(ch.get('no', i + 1) or (i + 1))
        title_toc = str(ch.get('title') or '')
        title_corr = str(ch.get('title_corrected') or '')
        title_for_lock = title_corr or title_toc

        hit = _find_tex_heading_near(
            tex_heading_events,
            chapter_no=no,
            chapter_title=title_for_lock,
            chapter_title_fallback=(title_corr if title_corr and title_corr != title_for_lock else ''),
            around=cur,
            window=win,
            prefer_side='either',
            promote_min_sim=promote_min_sim,
        )
        if not hit:
            items.append({'i': i, 'chapter_no': no, 'applied': False, 'reason': 'no_matching_heading_nearby'})
            continue

        hpos = int(hit['pos'])
        title_sim = float(hit.get('title_sim', 0.0) or 0.0)
        no_ok = False
        try:
            no_ok = (int(hit.get('event_no')) == no) or (int(hit.get('inferred_no')) == no)
        except Exception:
            no_ok = False

        # Hard guards: do not lock chapter boundaries to obvious section-like decimal headings.
        hit_title = str(hit.get('event_title') or hit.get('title') or '')
        if _is_section_like_integer_dot_title(hit_title):
            items.append({'i': i, 'chapter_no': no, 'applied': False, 'reason': 'section_like_heading', 'hit': hit})
            continue
        if any(abs(hpos - p) < cluster_sep for p in used_heading_pos):
            items.append({'i': i, 'chapter_no': no, 'applied': False, 'reason': 'reused_or_clustered_heading', 'hit': hit})
            continue
        if i > 0 and (hpos - int(out[i-1])) < min_prev_seg:
            items.append({'i': i, 'chapter_no': no, 'applied': False, 'reason': 'prev_segment_too_short', 'hit': hit})
            continue

        delta = hpos - cur
        absd = abs(delta)
        evidence_ok = bool(no_ok or title_sim >= min_title_sim)
        lead_applied = False
        status = ''
        if i < len(boundary_details) and isinstance(boundary_details[i], dict):
            lead_applied = bool(boundary_details[i].get('lead_applied'))
            status = str(boundary_details[i].get('status') or '')

        apply = False
        reason = ''
        if evidence_ok and absd <= max(1, win):
            if cur > hpos and (cur - hpos) <= force_after:
                apply = True; reason = 'after_target_heading'
            elif cur < hpos and (hpos - cur) <= force_before:
                apply = True; reason = 'before_target_heading'
            elif lead_applied or status == 'suspect':
                apply = True; reason = 'suspect_or_lead_shift'

        if apply:
            old = cur
            out[i] = hpos
            used_heading_pos.append(hpos)
            if i < len(boundary_details) and isinstance(boundary_details[i], dict):
                bd = boundary_details[i]
                bd['start_raw_before_tex_heading_lock'] = int(old)
                bd['start_raw'] = int(hpos)
                bd['tex_heading_lock_applied'] = True
                bd['tex_heading_lock_reason'] = reason
                bd['tex_heading_lock_hit'] = hit
                bd['method'] = str(bd.get('method') or '') + '+tex_heading_lock'
                bd['matched_tex_heading'] = str(hit_title)
            items.append({'i': i, 'chapter_no': no, 'applied': True, 'reason': reason, 'from': old, 'to': hpos, 'delta': hpos-old, 'hit': hit})
        else:
            items.append({'i': i, 'chapter_no': no, 'applied': False, 'reason': 'criteria_not_met', 'from': cur, 'hit': hit})

    # Optional reclaim: if the next chapter heading is inside the current span, reclaim boundary i+1 to it.
    if dual_enable and reclaim_enable and tex_heading_events and len(out) >= 2:
        reclaim_min_lead = int(getattr(cfg.align, 'tex_heading_lock_reclaim_min_lead_chars', 80) or 80)
        reclaim_min_sim = float(getattr(cfg.align, 'tex_heading_lock_reclaim_min_title_sim', 0.60) or 0.60)
        for i in range(len(out) - 1):
            nxt_info = chapters_anchored[i + 1] if (i + 1) < len(chapters_anchored) else None
            if not isinstance(nxt_info, dict):
                continue
            nxt_title = str(nxt_info.get('title_corrected') or nxt_info.get('title') or '')
            nxt_no = int(nxt_info.get('no', i + 2) or (i + 2))
            if not nxt_title:
                continue
            curr_start = int(out[i])
            curr_end = int(out[i + 1])
            if curr_end <= curr_start + max(reclaim_min_lead, 1):
                continue
            best = None
            for ev in tex_heading_events:
                pos = int(getattr(ev, 'start', -1) or -1)
                if pos <= curr_start + reclaim_min_lead or pos >= curr_end - mg:
                    continue
                ev_title = str(getattr(ev, 'title', '') or '')
                ev_kind = str(getattr(ev, 'kind', '') or '').lower()
                if _is_decimal_section_heading_title(ev_title) or _is_section_like_integer_dot_title(ev_title):
                    continue
                sim = title_similarity(normalize_title(nxt_title), normalize_title(ev_title))
                ev_no = getattr(ev, 'no', None)
                no_match = False
                try:
                    no_match = (ev_no is not None) and (int(ev_no) == nxt_no)
                except Exception:
                    no_match = False
                ev_chapterlike = _is_chapterlike_heading(ev_kind, ev_title)
                if not (no_match or sim >= reclaim_min_sim or (ev_chapterlike and sim >= min_title_sim)):
                    continue
                if any(abs(pos - p) < cluster_sep for p in used_heading_pos):
                    continue
                rank = (2.0 if no_match else 0.0) + float(sim)
                cand = (rank, -pos, pos, ev)
                if best is None or cand > best:
                    best = cand
            if best is not None:
                _rank, _npos, _pos, _ev = best
                old = int(out[i + 1])
                out[i + 1] = int(_pos)
                used_heading_pos.append(int(_pos))
                if (i + 1) < len(boundary_details) and isinstance(boundary_details[i + 1], dict):
                    bd = boundary_details[i + 1]
                    bd['reclaim_next_heading'] = True
                    bd['tex_heading_lock_applied'] = True
                    bd['start_raw'] = int(_pos)
                    bd['matched_tex_heading'] = str(getattr(_ev, 'title', '') or '')
                items.append({'i': i + 1, 'chapter_no': nxt_no, 'applied': True, 'reason': 'reclaim_next_heading', 'from': old, 'to': int(_pos), 'hit': {'event_title': str(getattr(_ev, 'title', '') or '')}})

    # Re-enforce monotonicity with a real minimum gap (avoid 1-char fragments).
    prev = -10**18
    for i in range(len(out)):
        need = prev + max(1, mg)
        if out[i] < need:
            old = int(out[i])
            out[i] = int(need)
            if i < len(boundary_details) and isinstance(boundary_details[i], dict):
                bd = boundary_details[i]
                bd['tex_heading_lock_monotone_bump'] = True
                bd['start_raw'] = int(out[i])
            items.append({'i': i, 'chapter_no': int(chapters_anchored[i].get('no', i + 1) or (i + 1)) if i < len(chapters_anchored) else i+1, 'applied': True, 'reason': 'monotone_gap_repair', 'from': old, 'to': int(out[i])})
        prev = int(out[i])

    return out, {'enabled': True, 'items': items, 'window': win, 'force_after': force_after, 'force_before': force_before, 'min_title_sim': min_title_sim, 'min_gap': mg}

def _tex_query_from_title(title: str, max_tokens: int = 10) -> str:
    t = normalize_title(title or "")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    toks = t.split(" ")
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]
    return " ".join(toks).strip()

def _map_tex_chapters_to_units(
    store: PDFUnitStore,
    raw_text: str,
    chapters: List[Dict[str, Any]],
    *,
    toc_end_unit: int,
    cfg,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Map TeX-derived chapters to PDF unit_start when printed_page labels are unavailable.

    Strategy (safe, non-regressive):
      1) Try title-to-page fuzzy matching using PDF text layer (body/full).
      2) If text layer is empty or no reliable match, fall back to proportional mapping by TeX char offsets.
    """
    fallbacks: List[str] = []
    n_units = int(getattr(store, "unit_count", 0) or 0)
    if n_units <= 0:
        return chapters, fallbacks

    page_text_cache: Dict[int, str] = {}

    def _get_page_text(u: int) -> str:
        if u in page_text_cache:
            return page_text_cache[u]
        try:
            unit = store.unit_ref(u)
            txt = store.extract_unit_text(unit, region="body") or ""
            if not txt.strip():
                txt = store.extract_unit_text(unit, region="full") or ""
        except Exception:
            txt = ""
        txtn = _norm_for_match(txt)
        page_text_cache[u] = txtn
        return txtn

    def _search_best_page(q: str, u0: int) -> Tuple[Optional[int], float]:
        if not q:
            return None, 0.0
        best_u = None
        best_s = 0.0
        step = int(getattr(cfg.align, "tex_page_search_stride", 3) or 3)
        u = max(int(u0), int(toc_end_unit))
        u_end = n_units - 1
        for uu in range(u, u_end + 1, step):
            pt = _get_page_text(uu)
            if not pt:
                continue
            s = float(partial_ratio(q, pt) / 100.0)
            if s > best_s:
                best_s, best_u = s, uu
                if best_s >= float(getattr(cfg.align, "tex_title_page_early_stop", 0.92) or 0.92):
                    break
        if best_u is None:
            return None, 0.0
        for uu in range(max(int(best_u) - step, int(toc_end_unit)), min(int(best_u) + step, u_end) + 1):
            pt = _get_page_text(uu)
            if not pt:
                continue
            s = float(partial_ratio(q, pt) / 100.0)
            if s > best_s:
                best_s, best_u = s, uu
        return best_u, best_s

    min_score = float(getattr(cfg.align, "tex_title_page_min_score", 0.72) or 0.72)
    prev_u = max(int(toc_end_unit), 0)
    any_text_layer = False

    for probe in range(min(prev_u + 10, n_units - 1), min(prev_u + 40, n_units - 1), 5):
        if _get_page_text(probe):
            any_text_layer = True
            break

    for ch in chapters:
        ch.setdefault("printed_page", "")
        ch_start_char = ch.get("_tex_char_start")
        ch["unit_start"] = None

        q = _tex_query_from_title(ch.get("title", ""), max_tokens=10)
        mapped = None
        score = 0.0
        if any_text_layer and q:
            mapped, score = _search_best_page(_norm_for_match(q), prev_u)
            if mapped is not None and mapped < prev_u:
                mapped = None

        if mapped is not None and score >= min_score:
            ch["unit_start"] = int(mapped)
            ch["_unit_start_method"] = "tex_title_match"
            ch["_unit_start_score"] = float(score)
            prev_u = int(mapped)
        else:
            if ch_start_char is None or not isinstance(ch_start_char, int) or ch_start_char < 0:
                ch_start_char = 0
            total = max(1, len(raw_text))
            frac = float(ch_start_char) / float(total)
            u_est = int(toc_end_unit + frac * max(1, (n_units - 1 - toc_end_unit)))
            u_est = max(prev_u, min(n_units - 1, u_est))
            ch["unit_start"] = int(u_est)
            ch["_unit_start_method"] = "tex_char_proportional"
            ch["_unit_start_score"] = float(score)
            prev_u = int(u_est)
            fallbacks.append("tex_unitstart_proportional")
            if any_text_layer and q:
                fallbacks.append("tex_unitstart_title_match_failed")

    return chapters, fallbacks

def _is_frontmatter_title(title: str) -> bool:
    """Heuristic filter for TeX/TOC headings that belong to front/back matter (目录/索引/参考文献等).

    Used only for fallbacks to avoid over-splitting.
    """
    t = normalize_title(title or "").lower().strip()
    if not t:
        return False
    # English/common
    kws = [
        "contents",
        "table of contents",
        "preface",
        "foreword",
        "acknowledg",
        "bibliography",
        "references",
        "reference",
        "index",
        "glossary",
        "notation",
        "symbols",
        "list of figures",
        "list of tables",
    ]
    for kw in kws:
        if kw in t:
            return True
    # Chinese common
    cjk_kws = ["目录", "前言", "序言", "序", "致谢", "参考文献", "索引", "符号表", "图目录", "表目录"]
    tt = title.strip()
    for kw in cjk_kws:
        if kw and kw in tt:
            return True
    return False


def _sanitize_tex_events_for_plan(
    raw_text: str,
    tex_events,
    *,
    cfg,
    logger: logging.Logger,
) -> Tuple[list, List[str]]:
    """Apply conservative controls to TeX-derived chapter-like events."""
    fallbacks: List[str] = []
    # filter front/back matter
    tex_events = [e for e in (tex_events or []) if not _is_frontmatter_title(getattr(e, "title", ""))]

    # Guard against over-granularity: too many events
    max_ch = int(getattr(cfg.align, "tex_fallback_max_chapters", 80) or 80)
    if len(tex_events) > max_ch:
        fallbacks.append("tex_plan_too_many_events")
        # Prefer explicit \chapter if available
        try:
            hs = parse_tex_headings(raw_text, dedup_gap_chars=int(getattr(cfg.align, 'tex_heading_dedup_gap_chars', 5000) or 5000))
            ch_only = [e for e in hs if getattr(e, "kind", "") == "chapter" and not _is_frontmatter_title(getattr(e, "title", ""))]
            if 2 <= len(ch_only) <= max_ch:
                tex_events = ch_only
                fallbacks.append("tex_plan_use_chapter_cmd_only")
            else:
                # Downsample by spacing to avoid 1-2 page micro-splits
                min_spacing = int(getattr(cfg.align, "tex_fallback_min_event_spacing_chars", 2000) or 2000)
                ds = []
                last = -10**18
                for e in tex_events:
                    st = int(getattr(e, "start", 0) or 0)
                    if st - last >= min_spacing:
                        ds.append(e)
                        last = st
                    if len(ds) >= max_ch:
                        break
                if len(ds) >= 2:
                    tex_events = ds
                    fallbacks.append("tex_plan_downsampled")
                else:
                    tex_events = []
        except Exception:
            pass

    return tex_events, fallbacks


def _backfill_units_by_title_match(
    store: PDFUnitStore,
    chapters: List[Dict[str, Any]],
    *,
    toc_end_unit: int,
    cfg,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Backfill missing unit_start using title-to-page fuzzy matching on PDF text layer.

    This runs ONLY for chapters whose unit_start is None, so it should not regress books with good page-label mapping.
    v12.4.9 hardening:
      - keep search monotone from the last processed chapter (not the global max existing unit)
      - bound search/fallback by the next known chapter unit when available
      - avoid end-of-book pileups that cause chapter drift and fragmentation
    """
    fallbacks: List[str] = []
    if not chapters:
        return chapters, fallbacks
    n_units = int(getattr(store, "unit_count", 0) or 0)
    if n_units <= 0:
        return chapters, fallbacks

    page_text_cache: Dict[int, str] = {}

    def _get_page_text(u: int) -> str:
        if u in page_text_cache:
            return page_text_cache[u]
        try:
            unit = store.unit_ref(u)
            txt = store.extract_unit_text(unit, region="body") or ""
            if not txt.strip():
                txt = store.extract_unit_text(unit, region="full") or ""
        except Exception:
            txt = ""
        txtn = _norm_for_match(txt)
        page_text_cache[u] = txtn
        return txtn

    any_text_layer = False
    probe_lo = max(int(toc_end_unit), 0)
    probe_hi = min(n_units - 1, probe_lo + 80)
    if probe_hi >= probe_lo:
        step_probe = 6 if (probe_hi - probe_lo) >= 12 else 1
        for probe in range(probe_lo, probe_hi + 1, step_probe):
            if _get_page_text(probe):
                any_text_layer = True
                break
    if not any_text_layer:
        fallbacks.append("unit_backfill_no_pdf_text_layer")
        return chapters, fallbacks

    stride = int(getattr(cfg.align, "unit_backfill_search_stride", 3) or 3)
    min_score = float(getattr(cfg.align, "unit_backfill_min_score", 0.74) or 0.74)
    early_stop = float(getattr(cfg.align, "unit_backfill_early_stop", 0.92) or 0.92)
    min_gap_units = max(1, int(getattr(cfg.pdf, "min_chapter_unit_gap", 2) or 2))

    def _next_known_upper(i: int) -> Optional[int]:
        for j in range(i + 1, len(chapters)):
            uj = chapters[j].get("unit_start")
            try:
                if uj is not None:
                    return int(uj)
            except Exception:
                continue
        return None

    def _search_best_page(q: str, u0: int, u_hi_hint: Optional[int]) -> Tuple[Optional[int], float]:
        if not q:
            return None, 0.0
        best_u = None
        best_s = 0.0
        lo = max(int(u0), int(toc_end_unit), 0)
        hi = n_units - 1
        if u_hi_hint is not None:
            hi = min(hi, int(u_hi_hint) - min_gap_units)
        if hi < lo:
            return None, 0.0
        for uu in range(lo, hi + 1, max(1, stride)):
            pt = _get_page_text(uu)
            if not pt:
                continue
            s = float(partial_ratio(q, pt) / 100.0)
            if s > best_s:
                best_s, best_u = s, uu
                if best_s >= early_stop:
                    break
        if best_u is None:
            return None, 0.0
        lo2 = max(lo, int(best_u) - max(1, stride))
        hi2 = min(hi, int(best_u) + max(1, stride))
        for uu in range(lo2, hi2 + 1):
            pt = _get_page_text(uu)
            if not pt:
                continue
            s = float(partial_ratio(q, pt) / 100.0)
            if s > best_s:
                best_s, best_u = s, uu
        return best_u, best_s

    prev_u = max(int(toc_end_unit), 0)
    for i, ch in enumerate(chapters):
        cur_u = ch.get("unit_start")
        if cur_u is not None:
            try:
                u_int = int(cur_u)
                if u_int <= prev_u and i > 0:
                    fallbacks.append("unit_backfill_existing_nonmonotone")
                prev_u = max(prev_u, u_int)
            except Exception:
                pass
            continue

        title = str(ch.get("title_corrected") or ch.get("title") or "")
        q = _tex_query_from_title(title, max_tokens=10)
        mapped = None
        score = 0.0
        nxt_u = _next_known_upper(i)
        if q:
            mapped, score = _search_best_page(_norm_for_match(q), prev_u + min_gap_units, nxt_u)
            if mapped is not None and mapped <= prev_u:
                mapped = None
            if mapped is not None and nxt_u is not None and mapped >= int(nxt_u):
                mapped = None

        if mapped is not None and score >= min_score:
            ch["unit_start"] = int(mapped)
            ch["_unit_start_method"] = "toc_title_match_backfill"
            ch["_unit_start_score"] = float(score)
            prev_u = int(mapped)
        else:
            nxt_u = _next_known_upper(i)
            lower = max(prev_u + min_gap_units, int(toc_end_unit))
            upper = (n_units - 1) if nxt_u is None else min(n_units - 1, int(nxt_u) - min_gap_units)
            if upper < lower:
                u_est = lower
            elif nxt_u is not None:
                # bounded midpoint placement inside the gap; later monotone repair will evenly spread runs
                u_est = lower + ((upper - lower) // 2)
            else:
                frac = float(i) / float(max(1, len(chapters) - 1))
                raw_est = int(toc_end_unit + frac * max(1, (n_units - 1 - toc_end_unit)))
                u_est = min(upper, max(lower, raw_est))
            u_est = max(lower, min(n_units - 1, int(u_est)))
            ch["unit_start"] = int(u_est)
            ch["_unit_start_method"] = "index_proportional_backfill"
            ch["_unit_start_score"] = float(score)
            prev_u = int(u_est)
            fallbacks.append("unit_backfill_proportional")

    return chapters, fallbacks


def _repair_chapter_unit_starts_monotone(
    chapters: List[Dict[str, Any]],
    *,
    toc_end_unit: int,
    unit_count: int,
    cfg,
    logger: logging.Logger,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Repair missing/non-monotone chapter unit_start values conservatively."""
    report: Dict[str, Any] = {"applied": False, "adjustments": []}
    if not chapters:
        return chapters, report
    n = len(chapters)
    n_units = max(1, int(unit_count or 0))
    min_gap_units = max(1, int(getattr(cfg.pdf, "min_chapter_unit_gap", 2) or 2))
    tail_step = max(4, int(getattr(cfg.align, "unit_backfill_tail_step_units", min_gap_units * 3) or (min_gap_units * 3)))

    vals: List[Optional[int]] = []
    prev = max(-1, int(toc_end_unit) - 1)
    for i, ch in enumerate(chapters):
        ui = ch.get("unit_start")
        try:
            u = int(ui) if ui is not None else None
        except Exception:
            u = None
        if u is not None and not (0 <= u < n_units):
            u = None
        if u is not None and u <= prev:
            report["adjustments"].append({"i": i, "type": "invalidate_nonmonotone", "before": u, "prev": prev})
            u = None
        vals.append(u)
        if u is not None:
            prev = u

    diffs = []
    pv = None
    for u in vals:
        if u is not None:
            if pv is not None and u > pv:
                diffs.append(u - pv)
            pv = u
    if diffs:
        diffs = sorted(diffs)
        tail_step = max(tail_step, int(diffs[len(diffs)//2]))

    def _set(idx: int, new_u: int, why: str):
        old = chapters[idx].get("unit_start")
        chapters[idx]["unit_start"] = int(new_u)
        chapters[idx]["_unit_start_repaired"] = True
        prev_m = str(chapters[idx].get("_unit_start_method") or "")
        chapters[idx]["_unit_start_method"] = (prev_m + "+monotone_repair") if prev_m else "monotone_repair"
        report["adjustments"].append({"i": idx, "type": why, "before": old, "after": int(new_u)})

    i = 0
    prev_anchor = max(int(toc_end_unit), 0)
    while i < n:
        if vals[i] is not None:
            prev_anchor = int(vals[i])
            i += 1
            continue
        j = i
        while j < n and vals[j] is None:
            j += 1
        next_anchor = int(vals[j]) if j < n and vals[j] is not None else None
        m = j - i
        cur_prev = int(prev_anchor)
        for k in range(m):
            idx = i + k
            rem = m - k
            lower = max(int(toc_end_unit), cur_prev + min_gap_units)
            if next_anchor is None:
                cand = min(n_units - 1, lower + max(0, tail_step - min_gap_units))
            else:
                max_allowed = next_anchor - min_gap_units * rem
                if max_allowed < lower:
                    cand = lower
                else:
                    room = max_allowed - lower
                    cand = lower + (room // max(1, rem))
            cand = max(0, min(n_units - 1, int(cand)))
            vals[idx] = cand
            _set(idx, cand, "fill_missing_or_nonmonotone")
            cur_prev = cand
        prev_anchor = cur_prev
        i = j

    prev = max(-1, int(toc_end_unit) - 1)
    for i, ch in enumerate(chapters):
        try:
            u = int(ch.get("unit_start"))
        except Exception:
            u = None
        if u is None:
            u = min(n_units - 1, max(0, prev + min_gap_units))
            _set(i, u, "final_fill_none")
        if u <= prev:
            u2 = min(n_units - 1, prev + min_gap_units)
            if u2 != u:
                _set(i, u2, "final_monotone_bump")
                u = u2
        prev = u

    if report["adjustments"]:
        report["applied"] = True
    return chapters, report


def _post_repair_bounds_min_gap(
    bounds: List[int],
    details: List[Dict[str, Any]],
    *,
    raw_len: int,
    cfg,
    logger: logging.Logger,
) -> Tuple[List[int], Dict[str, Any]]:
    """Post-check for boundary collapse after snap_boundaries and apply minimal repair.

    Only applies when many *unreliable* boundaries create very small segments, to avoid regressing good books.
    """
    b = [int(x) for x in (bounds or [])]
    report: Dict[str, Any] = {"applied": False, "adjustments": []}
    if not b or len(b) < 3:
        return b, report

    min_seg_chars = int(getattr(cfg.align, "min_segment_chars", 1500) or 1500)
    post_min_gap = int(getattr(cfg.align, "post_min_gap_chars", max(800, min_seg_chars // 3)) or max(800, min_seg_chars // 3))
    small_gap_ratio_th = float(getattr(cfg.align, "post_repair_small_gap_ratio", 0.15) or 0.15)
    small_gap_min_count = int(getattr(cfg.align, "post_repair_small_gap_min_count", 3) or 3)
    unreliable_conf = float(getattr(cfg.align, "post_repair_unreliable_conf", float(getattr(cfg.align, "interp_min_confidence", 0.78)) or 0.78))
    catastrophic_gap_chars = int(getattr(cfg.align, "catastrophic_gap_chars", 5) or 5)
    min_chapter_chars = int(getattr(cfg.align, "min_chapter_chars", min_seg_chars) or min_seg_chars)

    def _is_unreliable(i: int) -> bool:
        if i < 0 or i >= len(details):
            return True
        d = details[i] or {}
        status = str(d.get("status") or "").lower()
        method = str(d.get("method") or "").lower()
        conf = d.get("conf")
        suspect = d.get("suspect_reasons") or []
        if status in {"fallback_unreliable", "fallback"}:
            return True
        if method in {"llm_fallback", "fallback_est_no_opening"}:
            return True
        try:
            if conf is not None and float(conf) < unreliable_conf:
                return True
        except Exception:
            pass
        if isinstance(suspect, list) and len(suspect) >= 2:
            return True
        return False

    # identify small-gap boundaries
    # - catastrophic: always repair (prevents prev+1 chains)
    # - small/unreliable: repair only when it is frequent (to avoid regressing good books)
    small = []
    catastrophic = []
    for i in range(1, len(b)):
        gap = int(b[i]) - int(b[i - 1])
        if gap <= catastrophic_gap_chars:
            catastrophic.append(i)
        elif gap < post_min_gap and _is_unreliable(i):
            small.append(i)

    report["small_gap_count"] = int(len(small))
    report["catastrophic_gap_count"] = int(len(catastrophic))
    report["post_min_gap_chars"] = int(post_min_gap)
    report["min_chapter_chars"] = int(min_chapter_chars)
    report["catastrophic_gap_chars"] = int(catastrophic_gap_chars)
    report["raw_len"] = int(raw_len)

    trigger = (len(catastrophic) > 0) or (len(small) >= small_gap_min_count) or ((len(small) / float(max(1, len(b)))) >= small_gap_ratio_th)
    if not trigger:
        return b, report

    # apply minimal monotone + min-gap enforcement (only for marked boundaries)
    out = b[:]
    prev = out[0]
    for i in range(1, len(out)):
        if i in catastrophic:
            need = max(1, min_chapter_chars)
        elif i in small:
            need = max(1, post_min_gap)
        else:
            need = 1
        min_allowed = prev + need
        if out[i] < min_allowed:
            before = out[i]
            out[i] = min(min_allowed, raw_len)
            report["adjustments"].append({"i": i, "before": int(before), "after": int(out[i]), "need_gap": int(need)})
        prev = out[i]

    # final clamp
    out = [max(0, min(int(x), raw_len)) for x in out]
    for i in range(1, len(out)):
        if out[i] <= out[i-1]:
            out[i] = min(raw_len, out[i-1] + 1)

    report["applied"] = True
    report["bounds_before"] = b
    report["bounds_after"] = out
    return out, report


def _estimate_raw_from_unit(unit: int, *, unit_count: int, raw_len: int, points: List[Tuple[int, int]]) -> int:
    """Piecewise-linear estimate of raw offset from PDF unit index."""
    if unit_count <= 1:
        return 0
    u = int(unit)
    base_points = [(0, 0)] + sorted([(int(a), int(b)) for a, b in points if b >= 0], key=lambda t: t[0]) + [(unit_count - 1, raw_len)]
    # remove duplicates by unit keeping earliest
    dedup: List[Tuple[int, int]] = []
    seen = set()
    for uu, rr in base_points:
        if uu in seen:
            continue
        seen.add(uu)
        dedup.append((uu, rr))
    base_points = dedup

    # find bracket
    left = None
    right = None
    for uu, rr in base_points:
        if uu <= u:
            left = (uu, rr)
        if uu >= u and right is None:
            right = (uu, rr)
    if left is None:
        return 0
    if right is None:
        return raw_len
    if right[0] == left[0]:
        return int(left[1])
    t = (u - left[0]) / float(right[0] - left[0])
    return int(left[1] + t * (right[1] - left[1]))


def _vl_relocate_unit_for_chapter_start(
    store: PDFUnitStore,
    session_vl,
    cfg,
    *,
    chapter_no: int,
    chapter_title: str,
    base_unit: int,
    min_unit: int,
    max_unit: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Use the original PDF (VL) to re-check chapter start page for ambiguous boundaries.

    This is intentionally token-heavier than the default path and only runs on suspect/low-confidence
    boundaries. It scans a small neighborhood of PDF units and asks the VL model which page truly
    starts the target chapter, then returns the best unit to center the text-side fallback window.
    """
    report: Dict[str, Any] = {
        "enabled": False,
        "applied": False,
        "base_unit": int(base_unit),
        "best_unit": int(base_unit),
        "items": [],
    }
    try:
        enable = bool(getattr(cfg.align, "vl_boundary_relocate_enable", False))
    except Exception:
        enable = False
    if not enable or store is None or session_vl is None:
        return report
    report["enabled"] = True

    n_units = int(getattr(store, "unit_count", 0) or 0)
    if n_units <= 0:
        return report

    radius = int(getattr(cfg.align, "vl_boundary_relocate_radius_units", 4) or 4)
    step = max(1, int(getattr(cfg.align, "vl_boundary_relocate_step_units", 1) or 1))
    max_checks = max(1, int(getattr(cfg.align, "vl_boundary_relocate_max_checks", 11) or 11))
    min_score = float(getattr(cfg.align, "vl_boundary_relocate_min_score", 0.60) or 0.60)
    min_margin = float(getattr(cfg.align, "vl_boundary_relocate_min_margin", 0.03) or 0.03)
    prefer_true = bool(getattr(cfg.align, "vl_boundary_relocate_prefer_true_start", True))
    use_full = bool(getattr(cfg.align, "vl_boundary_relocate_use_full_region", True))

    lo = max(0, int(min_unit), int(base_unit) - radius)
    hi = min(n_units - 1, int(max_unit), int(base_unit) + radius)
    if hi < lo:
        return report

    # Probe base unit first, then expand outward by distance to reduce unnecessary calls.
    probe_units = list(range(lo, hi + 1, step))
    probe_units.sort(key=lambda u: (abs(int(u) - int(base_unit)), int(u)))
    probe_units = probe_units[:max_checks]

    items: List[Dict[str, Any]] = []
    for uu in probe_units:
        try:
            ref = store.unit_ref(int(uu))
            region = "full" if use_full else "body"
            img = store.render_unit(ref, dpi=store.dpi_low, region=region)
            out = vl_verify_chapter_start(
                session_vl,
                img,
                chapter_no=int(chapter_no),
                chapter_title=str(chapter_title or ""),
                cfg=cfg,
            )
        except Exception as ex:
            items.append({
                "unit": int(uu),
                "score": 0.0,
                "is_true_start": False,
                "heading_text": "",
                "error": str(ex),
                "score_adj": 0.0,
            })
            continue

        score = 0.0
        try:
            score = float((out or {}).get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        is_true = bool((out or {}).get("is_true_start", False))
        heading_text = str((out or {}).get("heading_text", "") or "")
        title_sim = _title_sim_loose(chapter_title, heading_text) if heading_text else 0.0
        # Small tie-break bonuses; do not overwhelm the VL score itself.
        score_adj = float(score) + (0.08 if is_true else 0.0) + 0.10 * float(title_sim)
        items.append({
            "unit": int(uu),
            "score": round(float(score), 4),
            "score_adj": round(float(score_adj), 4),
            "is_true_start": bool(is_true),
            "heading_text": heading_text[:300],
            "title_sim": round(float(title_sim), 4),
            "raw": out if isinstance(out, dict) else None,
        })

    report["items"] = items
    if not items:
        return report

    def _sort_key(it: Dict[str, Any]):
        # Prefer true-start positives, then adjusted score, then closeness to base.
        return (1 if bool(it.get("is_true_start")) else 0, float(it.get("score_adj", 0.0) or 0.0), -abs(int(it.get("unit", 0)) - int(base_unit)))

    ranked = sorted(items, key=_sort_key, reverse=True)
    best = ranked[0]
    base_rec = None
    for it in items:
        if int(it.get("unit", -1)) == int(base_unit):
            base_rec = it
            break
    if base_rec is None:
        base_rec = {"unit": int(base_unit), "score": 0.0, "score_adj": 0.0, "is_true_start": False}

    best_score = float(best.get("score", 0.0) or 0.0)
    best_adj = float(best.get("score_adj", 0.0) or 0.0)
    base_adj = float(base_rec.get("score_adj", 0.0) or 0.0)
    margin = best_adj - base_adj
    if best_score < min_score:
        return report
    if prefer_true and (not bool(best.get("is_true_start", False))) and bool(base_rec.get("is_true_start", False)):
        return report
    if int(best.get("unit", base_unit)) == int(base_unit):
        return report
    if margin < min_margin:
        return report

    report.update({
        "applied": True,
        "best_unit": int(best.get("unit", base_unit)),
        "best_score": round(float(best_score), 4),
        "best_score_adj": round(float(best_adj), 4),
        "base_score_adj": round(float(base_adj), 4),
        "margin": round(float(margin), 4),
        "best_is_true_start": bool(best.get("is_true_start", False)),
    })
    try:
        logger.info(
            "[BOUNDARY][VL] ch=%s relocate unit %s -> %s (score=%.2f, margin=%.2f)",
            chapter_no, base_unit, report["best_unit"], best_score, margin,
        )
    except Exception:
        pass
    return report


def _build_candidates_by_stride(match_text: str, lo: int, hi: int, opening_snippet: str, *,
                                stride: int, win: int, topk: int,
                                candidate_min_alnum_ratio: float = 0.0,
                                candidate_max_brace_ratio: float = 1.0) -> List[Tuple[int, str, float]]:
    """Return candidates as (offset, excerpt, score) within [lo,hi)."""
    candidates: List[Tuple[int, str, float]] = []
    lo = max(0, lo)
    hi = min(len(match_text), hi)
    if hi <= lo:
        return candidates

    for off in range(lo, hi, max(1, stride)):
        ex = match_text[off:min(len(match_text), off + win)]
        sc = _local_snippet_score(opening_snippet, ex)
        candidates.append((off, ex, sc))

    candidates.sort(key=lambda t: t[2], reverse=True)
    # Filter out low-quality excerpts (often TeX-heavy) to avoid misleading high local scores.
    # Keep the original candidates if filtering would leave too few options.
    def _ex_quality(ex: str):
        L = len(ex) or 1
        alnum = sum(1 for ch in ex if ch.isalnum())
        brace = sum(1 for ch in ex if ch in '{}')
        return alnum / L, brace / L

    try:
        min_alnum = float(candidate_min_alnum_ratio or 0.0)
        max_brace = float(candidate_max_brace_ratio or 1.0)
        filtered = []
        for off, ex, sc in candidates:
            ar, br = _ex_quality(ex)
            if ar >= min_alnum and br <= max_brace:
                filtered.append((off, ex, sc))
        # Require at least a small pool after filtering.
        if len(filtered) >= max(3, min(int(topk), 5)):
            candidates = filtered
    except Exception:
        pass

    # Keep diverse offsets (avoid near-duplicates)
    picked: List[Tuple[int, str, float]] = []
    for off, ex, sc in candidates:
        if not picked:
            picked.append((off, ex, sc))
        else:
            if all(abs(off - p[0]) >= win // 2 for p in picked):
                picked.append((off, ex, sc))
        if len(picked) >= topk:
            break
    return picked


def _llm_pick_boundary(session_llm, cfg, *, chapter_no: int, chapter_title: str,
                       opening_snippet: str, anchors: List[str],
                       candidates: List[Tuple[int, str, float]]) -> Dict[str, Any]:
    """Ask reasoning model to pick the correct boundary offset among candidates."""
    # keep prompt tight; we already pre-ranked candidates.
    header = (
        "You are locating the START boundary of a chapter within a long book text file.\n"
        "You will be given: (1) chapter metadata, (2) a short opening snippet extracted from the chapter's first page in the PDF,\n"
        "and (3) several candidate text excerpts with absolute character offsets into the FULL_TEXT.\n\n"
        "Choose the candidate whose excerpt best matches the PDF opening snippet and corresponds to the true start of the chapter.\n"
        "Return ONLY JSON: {best_offset:int|null, confidence:0..1, anchor_snippet:string|null, notes:string}.\n"
        "Rules:\n"
        "- best_offset must be an absolute character index into FULL_TEXT (same coordinate as candidates).\n"
        "- Prefer the TRUE CHAPTER START, not a subsection start. If a heading looks like 7.1 / 3.2 / 4.1, it is usually too late for the chapter boundary.\n"
        "- If the PDF snippet appears to start mid-chapter, still choose the candidate nearest the chapter-level heading (Chapter N / 第N章 / \\chapter{...} / integer-only chapter heading), not the later subsection.\n"
        "- Use CHAPTER_NO as a hard constraint when visible in the text. Title similarity is secondary and can be misleading because subsections share keywords.\n"
        "- anchor_snippet (if provided) must be a short exact substring (8-40 words) that occurs near the start and is unique within the local region.\n"
        "- If none are plausible, set best_offset=null and explain in notes.\n"
    )

    meta = (
        f"CHAPTER_NO: {chapter_no}\n"
        f"CHAPTER_TITLE: {chapter_title}\n"
        f"OPENING_SNIPPET_PDF: {opening_snippet[:1200]}\n"
    )
    if anchors:
        meta += "ANCHORS_PDF: " + " | ".join([a[:120] for a in anchors[:3]]) + "\n"

    cand_txt = []
    for i, (off, ex, sc) in enumerate(candidates, start=1):
        cand_txt.append(
            f"[CANDIDATE {i}] offset={off} local_score={sc:.2f}\n" + ex[:2200]
        )

    prompt = header + "\n" + meta + "\n" + "\n\n".join(cand_txt)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    llm_pick_max_tokens = int(getattr(cfg.align, "llm_pick_max_tokens", 900) or 900)
    return call_json(session_llm, cfg.models.llm_model, messages, max_tokens=llm_pick_max_tokens, temperature=0.0)


def _write_needs_review(book_out: Path, book_id: str, issues: List[Dict[str, Any]], artifacts: Dict[str, Any]):
    """Write review artifacts without polluting the正文 files.

    Scheme A: create a separate needs_review/ directory.
    """
    if not issues:
        return
    nr_dir = ensure_dir(book_out / "needs_review")
    js = {
        "book_id": book_id,
        "generated_at": now_ts(),
        "issues": issues,
        "artifacts": artifacts,
    }
    dump_json(nr_dir / "needs_review.json", js)
    lines = [
        "# Needs Review\n",
        f"- Book: {book_id}\n",
        f"- Generated: {js['generated_at']}\n",
        "\n## Issues\n",
    ]
    for it in issues:
        lines.append(f"- chapter {it.get('chapter_no')}: {it.get('type')} ({it.get('severity','')})\n")
        if it.get("notes"):
            lines.append(f"  - notes: {it['notes']}\n")
    (nr_dir / "needs_review.md").write_text("".join(lines), encoding="utf-8")
    # Expose paths via artifacts dict so batch summaries can link them.
    try:
        artifacts["needs_review_json"] = str(nr_dir / "needs_review.json")
        artifacts["needs_review_md"] = str(nr_dir / "needs_review.md")
    except Exception:
        pass

def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

def _write_marked(raw_text: str, boundaries: List[int], chapters: List[Dict[str, Any]], cfg, out_path: Path):
    parts = []
    n = len(chapters)
    ends = list(boundaries[1:] + [len(raw_text)])
    if n > 0:
        trimmed_end, trim_pat = _trim_backmatter_from_last_chapter(raw_text, int(boundaries[-1]), cfg)
        if trim_pat is not None:
            ends[-1] = min(int(ends[-1]), int(trimmed_end))
            try:
                chapters[-1]["boundary_repair_actions"] = list(dict.fromkeys((chapters[-1].get("boundary_repair_actions") or []) + ["trim_backmatter"]))
                chapters[-1]["backmatter_trim_keyword"] = str(trim_pat)
            except Exception:
                pass
    for i in range(n):
        no = int(chapters[i].get("no", i+1) or (i+1))
        title = str(chapters[i].get("title_corrected") or chapters[i].get("title") or "")
        marker = str(cfg.output.marker_tpl).format(no=no, title=title)
        parts.append(marker)
        parts.append(raw_text[boundaries[i]:ends[i]])
    out_path.write_text("".join(parts), encoding="utf-8")

def _scan_micro_fragment_segments(raw_text: str, boundaries: List[int], ends: List[int], *, max_len: int = 6) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, (s0, e0) in enumerate(zip(boundaries, ends)):
        try:
            s = int(s0); e = int(e0)
        except Exception:
            continue
        if e <= s:
            out.append({"i": i, "len": 0, "preview": "", "reason": "empty_segment"})
            continue
        seg = raw_text[s:e]
        stripped = seg.strip()
        if len(stripped) <= max_len:
            out.append({"i": i, "len": len(seg), "preview": stripped.replace("\n", "\\n"), "reason": "micro_segment"})
    return out


def _write_split(raw_text: str, boundaries: List[int], chapters: List[Dict[str, Any]], cfg, out_dir: Path):
    """Write per-chapter TXT files.

    Windows path length issues (MAX_PATH) are handled conservatively by shortening filenames.
    This does NOT change content; only output filenames are adjusted when needed.
    """
    ensure_dir(out_dir)
    n = len(chapters)
    ends = list(boundaries[1:] + [len(raw_text)])
    if n > 0:
        trimmed_end, trim_pat = _trim_backmatter_from_last_chapter(raw_text, int(boundaries[-1]), cfg)
        if trim_pat is not None:
            ends[-1] = min(int(ends[-1]), int(trimmed_end))
            try:
                chapters[-1]["boundary_repair_actions"] = list(dict.fromkeys((chapters[-1].get("boundary_repair_actions") or []) + ["trim_backmatter"]))
                chapters[-1]["backmatter_trim_keyword"] = str(trim_pat)
            except Exception:
                pass

    is_windows = (os.name == "nt")
    max_path = int(getattr(getattr(cfg, "runtime", cfg), "max_windows_path_len", 245) or 245)
    title_ml_default = int(getattr(getattr(cfg, "runtime", cfg), "chapter_filename_max_len", 80) or 80)

    def _make_path(no: int, title: str, ml: int) -> Path:
        comp = safe_output_filename(title, max_len=max(12, int(ml)), add_hash=True)
        fn = f"{no:02d}_{comp}.txt"
        return out_dir / fn

    for i in range(n):
        no = int(chapters[i].get("no", i + 1) or (i + 1))
        s = int(boundaries[i])
        try:
            chapters[i]["segment_start_title"] = _extract_segment_start_tex_chapter_title(raw_text, s)
        except Exception:
            chapters[i]["segment_start_title"] = ""
        title = _resolve_output_chapter_title(chapters[i], cfg).strip() or f"chapter_{no}"
        e = int(ends[i])

        ml = title_ml_default
        path = _make_path(no, title, ml)

        if is_windows:
            # Soft loop to shorten to avoid MAX_PATH errors.
            for _ in range(12):
                if len(str(path)) <= max_path:
                    break
                ml = max(12, ml - 8)
                path = _make_path(no, title, ml)

            if len(str(path)) > max_path:
                # Last resort: use hash-only filename component.
                h = hashlib.sha1(title.encode("utf-8", errors="ignore")).hexdigest()[:12]
                path = out_dir / f"{no:02d}_{h}.txt"

        try:
            path.write_text(raw_text[s:e], encoding="utf-8")
        except Exception as ex:
            # Retry once with hash-only filename (common on Windows when path is still too long).
            try:
                h = hashlib.sha1(title.encode("utf-8", errors="ignore")).hexdigest()[:12]
                alt = out_dir / f"{no:02d}_{h}.txt"
                alt.write_text(raw_text[s:e], encoding="utf-8")
                chapters[i]["_chapter_file"] = str(alt)
                chapters[i]["_chapter_file_fallback"] = "hash_only"
            except Exception:
                # Propagate to upper layer: caller will record and write needs_review/summary.
                raise


def process_one_book(session, cfg_path: Path, pdf_path: Path, text_path: Path) -> Dict[str, Any]:
    """
    Run closed-loop pipeline for one (PDF, TXT/TeX) pair.
    Returns a summary dict (success/fallbacks/paths).
    """
    cfg = load_config(cfg_path)
    out_root = Path(cfg.runtime.out_dir)
    # Use a Windows-safe, length-bounded directory name (avoid MAX_PATH issues on deep trees).
    max_book_dir_len = int(getattr(cfg.runtime, "max_book_dir_len", 120) or 120)
    book_id = safe_output_filename(pdf_path.stem, max_len=max_book_dir_len, add_hash=True)
    book_out = ensure_dir(out_root / book_id)
    cache_dir = ensure_dir(book_out / cfg.runtime.cache_dirname)
    logs_dir = ensure_dir(book_out / cfg.runtime.logs_dirname)
    log_path = logs_dir / "run.log"
    logger = _setup_logger(log_path)

    summary = {
        "book_id": book_id,
        "pdf": str(pdf_path),
        "text": str(text_path),
        "out_dir": str(book_out.resolve()),
        "success": False,
        "fallbacks": [],
        "artifacts": {},
    }


    # Runtime metadata propagated by run.py (env vars) for auditability.
    # These fields are optional and do not affect splitting logic.
    run_id = os.environ.get("PDFTXTALIGN_RUN_ID")
    config_hash = os.environ.get("PDFTXTALIGN_CONFIG_HASH")
    code_hash = os.environ.get("PDFTXTALIGN_CODE_HASH")
    code_version = os.environ.get("PDFTXTALIGN_CODE_VERSION")
    code_version_source = os.environ.get("PDFTXTALIGN_CODE_VERSION_SOURCE")
    config_path = os.environ.get("PDFTXTALIGN_CONFIG_PATH")
    cwd = os.environ.get("PDFTXTALIGN_CWD")
    pid = os.environ.get("PDFTXTALIGN_PID")
    if run_id:
        summary["run_id"] = str(run_id)
    if config_hash:
        summary["config_hash"] = str(config_hash)
    if code_hash:
        summary["code_hash"] = str(code_hash)
    if code_version:
        summary["code_version"] = str(code_version)
    if code_version_source:
        summary["code_version_source"] = str(code_version_source)
    logger.info(
        "[RUN] run_id=%s code_version=%s code_hash=%s config_hash=%s pid=%s cwd=%s",
        run_id, code_version, code_hash, config_hash, pid, cwd,
    )
    try:
        # Store run metadata inside the per-book cache for evidence chain closure.
        # This is deliberately redundant with outputs/_batch/<run_id>/run_manifest.json.
        dump_json(cache_dir / "meta.json", {
            "book_id": book_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": str(run_id) if run_id else None,
            "config_hash": str(config_hash) if config_hash else None,
            "config_path": str(config_path) if config_path else str(cfg_path),
            "code_hash": str(code_hash) if code_hash else None,
            "code_version": str(code_version) if code_version else None,
            "code_version_source": str(code_version_source) if code_version_source else None,
            "cwd": str(cwd) if cwd else None,
            "pid": int(pid) if (pid and str(pid).isdigit()) else None,
            "cfg_path": str(cfg_path),
        })
        summary["artifacts"]["meta"] = str(cache_dir / "meta.json")
    except Exception:
        pass
    # Accumulate pipeline-level fallback flags across stages (must exist even if early failures happen)
    fallbacks: List[str] = []

    # Source of the chapter plan ('toc' | 'tex'). Default to 'toc' for normal path.
    plan_source: str = "toc"


    store = None
    try:
        logger.info(f"=== Book: {book_id} ===")
        logger.info(f"PDF: {pdf_path}")
        logger.info(f"TEXT: {text_path}")

        # Load text
        raw_text = text_path.read_text(encoding="utf-8", errors="ignore")

        # PDF unit store
        store = PDFUnitStore(
            pdf_path=pdf_path,
            cache_dir=ensure_dir(cache_dir / "images"),
            dpi_low=int(cfg.pdf.dpi_low),
            dpi_high=int(cfg.pdf.dpi_high),
            two_up_threshold=float(cfg.pdf.double_page_aspect_ratio_threshold),
            force_layout=(str(getattr(cfg.pdf, "force_layout", "") or "").strip() or None),
            force_gutter_x=(int(getattr(cfg.pdf, "force_gutter_x")) if getattr(cfg.pdf, "force_gutter_x", None) is not None else None),
            gutter_detect=bool(getattr(cfg.pdf, "double_page_gutter_detect", True)),
            gutter_white_frac_min=float(getattr(cfg.pdf, "double_page_gutter_white_frac_min", 0.85) or 0.85),
            gutter_score_min=float(getattr(cfg.pdf, "double_page_gutter_score_min", 0.25) or 0.25),
        )

        summary["artifacts"]["layout"] = store.layout_info.__dict__
        logger.info(f"[PDF] layout={store.layout_info.layout}, pdf_pages={store.pdf_page_count}, units={store.unit_count}")

        # TOC range
        toc_range, toc_scan = find_toc_range(store, session, cfg, cache_dir, logger)
        summary["artifacts"]["toc_scan"] = str(cache_dir / "toc_scan.json")
        fallbacks.extend(toc_scan.get("fallbacks", []) if isinstance(toc_scan, dict) else [])
        if toc_range is None:
            # no TOC -> fallback: still try splitting via TeX headings if available, else abort
            fallbacks.append("no_toc_range")
            chapters = []
        else:
            logger.info(f"[TOC] range units [{toc_range.start_unit},{toc_range.end_unit}]")
            toc = extract_and_parse_toc(store, session, session, cfg, toc_range, cache_dir, logger)
            # Expose TOC artifacts for batch-level audit/triage
            try:
                for _k, _p in {
                    'toc_markdown': cache_dir / 'toc_markdown.md',
                    'toc_attempts': cache_dir / 'toc_attempts.json',
                    'toc_parse': cache_dir / 'toc_parse.json',
                    'toc_lang_hint': cache_dir / 'toc_lang_hint.json',
                }.items():
                    if _p.exists():
                        summary['artifacts'][_k] = str(_p)
            except Exception:
                pass

            chapters = toc.get("chapters", []) if isinstance(toc, dict) else []
            # Language hint for page label crops (heuristic from TOC markdown)
            try:
                md_path = cache_dir / "toc_markdown.md"
                md_txt = md_path.read_text(encoding="utf-8", errors="ignore") if md_path.exists() else ""
                if md_txt:
                    cjk = sum(1 for ch in md_txt if "\u4e00" <= ch <= "\u9fff")
                    ratio = cjk / float(max(1, len(md_txt)))
                    hint = "zh" if (cjk >= 40 and ratio >= 0.02) else "en"
                    setattr(cfg.pdf, "page_label_language_hint", hint)
                    dump_json(cache_dir / "toc_lang_hint.json", {"hint": hint, "cjk": cjk, "ratio": ratio})
            except Exception:
                pass
            if toc.get("_fallback"):
                fb = toc.get("_fallback")
                if fb:
                    if isinstance(fb, str) and fb.lower() not in {"true", "1", "yes"}:
                        fallbacks.append(f"toc_fallback:{fb}")
                    else:
                        reason = toc.get("_fallback_reason") or "toc_fallback"
                        fallbacks.append(f"toc_fallback:{reason}")
            # Do not re-apply a fixed min_required gate here.
            # TOC validation (incl. small-book adaptive thresholds) is handled inside toc.extract_and_parse_toc.
            if not chapters:
                fallbacks.append("toc_chapters_insufficient")
                chapters = []

        # If no chapters from TOC, attempt TeX chapters as a fallback "chapter plan"
        if not chapters:
            tex_events = parse_tex_chapters(
                raw_text,
                min_chapters=int(getattr(cfg.align, 'tex_heading_min_chapters', 3) or 3),
                dedup_gap_chars=int(getattr(cfg.align, 'tex_heading_dedup_gap_chars', 5000) or 5000),
                toc_titles=([str((ch or {}).get('title') or '') for ch in (chapters or [])] if bool(getattr(cfg.align, 'tex_toc_title_promotion_enable', True)) else None),
                toc_title_promotion_min_sim=float(getattr(cfg.align, 'tex_toc_title_promotion_min_sim', 0.72) or 0.72),
            )
            tex_events, fb_tex = _sanitize_tex_events_for_plan(raw_text, tex_events, cfg=cfg, logger=logger)
            fallbacks.extend(fb_tex)

            # If TeX parsing didn't yield enough, try plain "CHAPTER n" regex on the TeX/plain text.
            if not tex_events or len(tex_events) < 2:
                try:
                    plain_events = parse_plain_chapters(raw_text) or []
                    plain_events = [e for e in plain_events if not _is_frontmatter_title(getattr(e, "title", ""))]
                    if plain_events and len(plain_events) >= 2:
                        tex_events = plain_events
                        fallbacks.append("use_plain_chapter_regex_as_plan")
                except Exception:
                    pass

            if tex_events and len(tex_events) >= 2:
                fallbacks.append("use_tex_as_chapter_plan")
                plan_source = "tex"
                chapters = [{"no": i+1, "title": e.title, "printed_page": "", "_tex_char_start": int(getattr(e, "start", 0) or 0)} for i, e in enumerate(tex_events)]
            else:
                # Emit a minimal, auditable needs_review bundle before failing hard.
                try:
                    issues = [{
                        "code": "no_chapter_plan",
                        "message": "No usable TOC and no TeX chapter-like headings; cannot build a chapter plan.",
                        "toc_range": [int(toc_range.start_unit), int(toc_range.end_unit)] if toc_range else None,
                        "tex_events": int(len(tex_events) if tex_events else 0),
                        "fallbacks": list(fallbacks),
                    }]
                    # book_id is the stable output identifier; avoid undefined vars on no-plan failures.
                    _write_needs_review(book_out, book_id, issues, summary.get("artifacts") or {})
                except Exception:
                    pass
                raise RuntimeError("No usable TOC and no TeX chapters; cannot build a chapter plan.")

        # Map chapter starts to unit indices (needs toc_end_unit)
        toc_end_unit = toc_range.end_unit if toc_range else 0
        locator = PageLocator(store, session, cfg, cache_dir, logger)
        if plan_source == "tex":
            chapters_mapped, fb2 = _map_tex_chapters_to_units(store, raw_text, chapters, toc_end_unit=toc_end_unit, cfg=cfg, logger=logger)
            fallbacks.extend(fb2)
        else:
            chapters_mapped, fb2 = locator.map_chapter_start_units(chapters, toc_end_unit=toc_end_unit)
            fallbacks.extend(fb2)

        # Backfill missing unit_start where printed-page mapping failed (non-regressive)
        try:
            chapters_mapped, fb3 = _backfill_units_by_title_match(store, chapters_mapped, toc_end_unit=toc_end_unit, cfg=cfg, logger=logger)
            fallbacks.extend(fb3)
        except Exception:
            pass

        # Monotone repair for mapped chapter starts (v12.4.9 hardening)
        try:
            chapters_mapped, unit_monotone_report = _repair_chapter_unit_starts_monotone(
                chapters_mapped,
                toc_end_unit=toc_end_unit,
                unit_count=int(getattr(store, 'unit_count', 0) or 0),
                cfg=cfg,
                logger=logger,
            )
            if unit_monotone_report.get('applied'):
                fallbacks.append('chapter_unit_monotone_repair')
                dump_json(cache_dir / 'chapter_unit_monotone_repair.json', unit_monotone_report)
                summary['artifacts']['chapter_unit_monotone_repair'] = str(cache_dir / 'chapter_unit_monotone_repair.json')
        except Exception:
            unit_monotone_report = {'applied': False, 'error': 'unit_monotone_repair_exception'}

        # Derive unit_end by next start
        for i in range(len(chapters_mapped)):
            u0 = chapters_mapped[i].get("unit_start")
            u1 = chapters_mapped[i+1].get("unit_start") if i+1 < len(chapters_mapped) else None
            chapters_mapped[i]["unit_end"] = (u1 - 1) if (u0 is not None and u1 is not None) else None

        dump_json(cache_dir / "chapter_plan.json", chapters_mapped)
        summary["artifacts"]["chapter_plan"] = str(cache_dir / "chapter_plan.json")

        # Anchor extraction to improve alignment and corrected titles
        chapters_anchored = extract_anchors_for_chapters(store, session, cfg, chapters_mapped, cache_dir, logger)
        dump_json(cache_dir / "chapters_with_anchors.json", chapters_anchored)
        summary["artifacts"]["chapters_with_anchors"] = str(cache_dir / "chapters_with_anchors.json")

        # Determine boundaries in text
        is_tex = text_path.suffix.lower() == ".tex"
        # For TeX inputs, use a same-length shadow text for matching/verification (math/macros blanked)
        match_text = build_tex_shadow_same_len(raw_text) if is_tex else raw_text
        tex_heading_events_for_lock: List[Any] = []
        if is_tex and bool(getattr(cfg.align, "tex_heading_lock_enable", True)):
            try:
                toc_titles_for_promo = [
                    str((c or {}).get("title_corrected") or (c or {}).get("title") or "")
                    for c in (chapters_anchored or [])
                ]
                tex_heading_events_for_lock = parse_tex_chapters(
                    raw_text,
                    toc_titles=toc_titles_for_promo,
                    min_chapters=int(getattr(cfg.align, "tex_heading_min_chapters", 3) or 3),
                    dedup_gap_chars=int(getattr(cfg.align, "tex_heading_dedup_gap_chars", 5000) or 5000),
                    toc_title_promotion_min_sim=float(getattr(cfg.align, "tex_toc_title_promotion_min_sim", 0.72) or 0.72),
                ) or []
                if not tex_heading_events_for_lock:
                    tex_heading_events_for_lock = parse_tex_headings(
                        raw_text,
                        dedup_gap_chars=int(getattr(cfg.align, "tex_heading_dedup_gap_chars", 5000) or 5000),
                    ) or []
            except Exception:
                tex_heading_events_for_lock = []

        b_fallbacks: List[str] = []
        issues: List[Dict[str, Any]] = []

        # 1) compute primary boundaries (TeX headings) if enabled
        bds_primary = []
        if is_tex and bool(cfg.align.prefer_tex_headings):
            bds_primary, fb_primary = boundaries_from_tex_headings(raw_text, chapters_anchored, cfg, logger)
            b_fallbacks.extend(fb_primary)

        # 2) compute secondary boundaries (anchor search) as a safety net
        bds_secondary, fb_secondary = boundaries_by_anchor_search(
            session,
            text_path,
            raw_text,
            chapters_anchored,
            cfg,
            cache_dir,
            logger,
            unit_count=int(store.unit_count),
        )
        b_fallbacks.extend(fb_secondary)

        by_no_primary = {b.chapter_no: b for b in bds_primary} if bds_primary else {}
        by_no_secondary = {b.chapter_no: b for b in bds_secondary} if bds_secondary else {}

        # 3) closed-loop boundary selection with per-chapter fallback (LLM decides cut point)
        raw_len = len(raw_text)
        unit_count = int(store.unit_count)
        min_conf = float(getattr(cfg.align, "min_accept_confidence", 0.72))
        snippet_min = float(getattr(cfg.align, "snippet_min_score", 0.35))
        fb_window = int(getattr(cfg.align, "fallback_window_chars", 200000))
        cand_stride = int(getattr(cfg.align, "candidate_stride_chars", 800))
        cand_win = int(getattr(cfg.align, "candidate_window_chars", 4000))
        cand_topk = int(getattr(cfg.align, "topk_candidates", getattr(cfg.align, "candidate_topk", 8)))
        min_seg_chars = int(getattr(cfg.align, "min_segment_chars", 1500))
        min_chapter_chars = int(getattr(cfg.align, "min_chapter_chars", min_seg_chars) or min_seg_chars)
        catastrophic_gap_chars = int(getattr(cfg.align, "catastrophic_gap_chars", 5) or 5)
        interp_min_conf = float(getattr(cfg.align, "interp_min_confidence", 0.78))

        final_bounds: List[int] = []
        boundary_details: List[Dict[str, Any]] = []
        prev = -1

        # Points for unit->raw interpolation (we add as we confirm boundaries)
        interp_points: List[Tuple[int, int]] = []

        for idx, ch in enumerate(chapters_anchored):
            no = int(ch.get("no", idx + 1) or (idx + 1))
            title = str(ch.get("title_corrected") or ch.get("title") or "").strip()
            opening = str(ch.get("opening_snippet") or "").strip()
            anchors = ch.get("anchors") or []
            if isinstance(anchors, str):
                anchors = [anchors]
            anchors = [a for a in anchors if isinstance(a, str) and a.strip()]

            primary = by_no_primary.get(no)
            secondary = by_no_secondary.get(no)

            # Choose best initial boundary candidate (by conf)
            candidates_init: List[Tuple[str, int, float]] = []
            if primary and int(primary.start_raw) >= 0:
                candidates_init.append((primary.method, int(primary.start_raw), float(primary.conf or 0.0)))
            if secondary and int(secondary.start_raw) >= 0:
                candidates_init.append((secondary.method, int(secondary.start_raw), float(secondary.conf or 0.0)))
            candidates_init.sort(key=lambda t: t[2], reverse=True)
            chosen_method, chosen, chosen_conf, candidate_policy_flags = _pick_initial_boundary_candidate(candidates_init, cfg)

            suspect = False
            suspect_reasons: List[str] = []
            local = 0.0
            toc_like = 0.0
            if chosen is None:
                suspect = True
                suspect_reasons.append("no_candidate")
            else:
                if chosen <= prev:
                    suspect = True
                    suspect_reasons.append("non_monotonic")

                # Evaluate local snippet consistency and TOC-likeness of the chosen boundary.
                lead_offset = None
                lead_score = None
                if opening:
                    lead_scan_chars = int(getattr(cfg.align, "lead_scan_chars", 20000) or 20000)
                    lead_win = match_text[chosen: min(raw_len, chosen + max(lead_scan_chars, 8000))]
                    local, lead_offset, lead_score = _local_snippet_alignment(opening, lead_win, scan_chars=lead_scan_chars)
                toc_like = _toc_likeness((raw_text if is_tex else match_text)[chosen: min(raw_len, chosen + 16000)])

                # Lead-offset guard: partial_ratio can be high even if the best match starts far away.
                lead_applied = False
                lead_before = lead_offset
                lead_score_before = lead_score
                lead_max = int(getattr(cfg.align, "lead_max_offset", 600) or 600)
                lead_strategy = str(getattr(cfg.align, "lead_offset_strategy", "shift") or "shift").strip().lower()
                snippet_soft_min = float(getattr(cfg.align, "snippet_soft_min_score", 0.15) or 0.15)
                if lead_offset is not None and int(lead_offset) > int(lead_max) and float(local) >= float(snippet_soft_min):
                    # Before shifting forward, check whether a strong TeX chapter heading for THIS chapter
                    # already exists near the proposed boundary. If so, lock to that heading instead of
                    # blindly following a late snippet match (classic "next chapter heading left in previous file" bug).
                    locked_on_heading = False
                    if is_tex and bool(getattr(cfg.align, "lead_shift_protect_tex_heading", True)):
                        try:
                            protect_win = int(getattr(cfg.align, "lead_shift_heading_protect_window_chars", 120000) or 120000)
                            hit = _find_tex_heading_near(
                                tex_heading_events_for_lock,
                                chapter_no=no,
                                chapter_title=title,
                                around=int(chosen),
                                window=min(max(int(lead_offset) + 2000, 4000), protect_win),
                                prefer_side='either',
                            )
                            if hit is not None:
                                chosen = int(hit.get('pos'))
                                chosen_method = f"{chosen_method}+lead_heading_lock"
                                chosen_conf = float(min(float(chosen_conf), 0.90))
                                lead_applied = False
                                locked_on_heading = True
                                if opening:
                                    lead_scan_chars = int(getattr(cfg.align, "lead_scan_chars", 20000) or 20000)
                                    lead_win2 = match_text[chosen: min(raw_len, chosen + max(lead_scan_chars, 8000))]
                                    local, lead_offset, lead_score = _local_snippet_alignment(opening, lead_win2, scan_chars=lead_scan_chars)
                                toc_like = _toc_likeness((raw_text if is_tex else match_text)[chosen: min(raw_len, chosen + 16000)])
                        except Exception:
                            locked_on_heading = False
                    if (not locked_on_heading) and lead_strategy == "shift":
                        new_chosen = int(chosen) + int(lead_offset)
                        if 0 <= new_chosen < raw_len:
                            chosen = int(new_chosen)
                            chosen_method = f"{chosen_method}+lead_shift"
                            chosen_conf = float(min(float(chosen_conf), 0.88))
                            lead_applied = True
                            if opening:
                                lead_scan_chars = int(getattr(cfg.align, "lead_scan_chars", 20000) or 20000)
                                lead_win2 = match_text[chosen: min(raw_len, chosen + max(lead_scan_chars, 8000))]
                                local, lead_offset, lead_score = _local_snippet_alignment(opening, lead_win2, scan_chars=lead_scan_chars)
                            toc_like = _toc_likeness((raw_text if is_tex else match_text)[chosen: min(raw_len, chosen + 16000)])
                    elif not locked_on_heading:
                        suspect = True
                        suspect_reasons.append(f"lead_offset_too_far:{int(lead_offset)}")

                # Hard floor: extremely low local snippet match is a strong indicator of a wrong boundary
                # (often due to TeX-heavy match_text or duplicated headings). Force fallback in this case.
                hard_low = float(getattr(cfg.align, 'hard_low_local_snippet_score', 0.0) or 0.0)
                if opening and hard_low > 0.0 and float(local or 0.0) < hard_low:
                    suspect = True
                    suspect_reasons.append(f'hard_low_local_snippet:{float(local or 0.0):.2f}')
                    b_fallbacks.append(f'hard_low_local_snippet_ch{no}')

                # Accept if either model confidence OR local evidence is strong enough.
                combined = max(float(chosen_conf or 0.0), float(local or 0.0))
                if combined < min_conf:
                    suspect = True
                    suspect_reasons.append(f"low_combined_conf:{combined:.2f}")
                if prev >= 0 and (chosen - prev) < min_seg_chars:
                    suspect = True
                    suspect_reasons.append("too_short_segment")
                    b_fallbacks.append(f"too_short_segment_ch{no}")


                # Hard guard: extremely low local snippet score indicates the candidate is likely wrong
                # (common when the chosen offset lands inside TeX artifacts or unrelated text).
                hard_low = float(getattr(cfg.align, 'hard_low_local_snippet_score', 0.0) or 0.0)
                if opening and hard_low > 0 and local < hard_low:
                    suspect = True
                    suspect_reasons.append(f"hard_low_local_snippet:{local:.2f}")
                    b_fallbacks.append(f"hard_low_local_snippet_ch{no}")

                # Only promote low-local into *hard* suspect when there are additional red flags.
                if opening and local < snippet_min:
                    if toc_like >= 0.55 or float(chosen_conf or 0.0) < (min_conf - 0.15):
                        suspect = True
                        suspect_reasons.append(f"low_local_snippet:{local:.2f}")
                        b_fallbacks.append(f"low_local_snippet_ch{no}")
                    else:
                        # record as soft signal (do not force fallback)
                        b_fallbacks.append(f"soft_low_local_snippet_ch{no}")

                if toc_like >= 0.70:
                    suspect = True
                    suspect_reasons.append(f"toc_like:{toc_like:.2f}")

            if not suspect:
                final = int(chosen)
                boundary_details.append({
                    "chapter_no": no,
                    "start_raw": final,
                    "method": chosen_method,
                    "conf": chosen_conf,
                    "local_snippet_score": local,
                    "lead_offset": (int(lead_offset) if lead_offset is not None else None),
                    "lead_score": (float(lead_score) if lead_score is not None else None),
                    "lead_max_offset": int(lead_max),
                    "lead_strategy": str(lead_strategy),
                    "lead_applied": bool(lead_applied),
                    "lead_offset_before": (int(lead_before) if lead_before is not None else None),
                    "lead_score_before": (float(lead_score_before) if lead_score_before is not None else None),
                    "toc_likeness": toc_like,
                    "candidates_init": candidates_init[:5],
                    "candidates_init_full": candidates_init,
                    "candidate_policy_flags": candidate_policy_flags,
                    "status": "ok",
                })
                final_bounds.append(final)
                prev = final
                u = ch.get("unit_start")
                if u is not None and float(chosen_conf) >= interp_min_conf:
                    interp_points.append((int(u), final))
                elif u is not None:
                    issues.append({
                        "chapter_no": no,
                        "type": "interp_anchor_skipped_low_conf",
                        "severity": "low",
                        "notes": f"conf={float(chosen_conf):.2f} < interp_min_conf={interp_min_conf:.2f}",
                    })
                continue

            # --- Per-chapter fallback: LLM decides boundary within an estimated window ---
            u = ch.get("unit_start")
            unit_idx = int(u) if u is not None else int(unit_count * (idx / max(1, len(chapters_anchored))))

            # PDF-guided calibration (quality-first): support text-layer probe + VL relocate with retries.
            vl_unit_reloc = None
            pdf_text_probe = None
            try:
                prev_u_bound = 0
                next_u_bound = max(0, unit_count - 1)
                try:
                    if idx > 0:
                        pu = chapters_anchored[idx - 1].get("unit_start")
                        if pu is not None:
                            prev_u_bound = max(prev_u_bound, int(pu) + 1)
                except Exception:
                    pass
                try:
                    if idx + 1 < len(chapters_anchored):
                        nu = chapters_anchored[idx + 1].get("unit_start")
                        if nu is not None:
                            next_u_bound = min(next_u_bound, int(nu))
                except Exception:
                    pass

                mode = str(getattr(cfg.align, "vl_boundary_relocate_mode", "suspect") or "suspect").lower()
                # anchor_unit may be absent for some books (or older artifacts). Do not reference a
                # local variable before assignment; fall back to unit_start when anchor_unit is missing.
                anchor_unit = ch.get("anchor_unit")
                if anchor_unit is None:
                    anchor_unit = ch.get("unit_start")
                low_conf_mode = (anchor_unit is None) or (
                    float(ch.get("anchor_confidence", 0.0) or 0.0)
                    < float(getattr(cfg.align, "vl_boundary_verify_min_score", 0.78))
                )
                need_vl = bool(getattr(cfg.align, "vl_boundary_relocate_enable", False)) and (
                    mode == "always" or (mode == "low_conf" and low_conf_mode) or mode == "suspect"
                )
                if bool(getattr(cfg.align, "vl_boundary_verify_after_lock", False)):
                    need_vl = bool(getattr(cfg.align, "vl_boundary_relocate_enable", False)) and (mode in {"always", "low_conf", "suspect"})

                # For text-layer PDFs, try a cheap text probe first.
                if need_vl and bool(getattr(cfg.align, "pdf_text_probe_enable", True)):
                    pdf_text_probe = _pdf_text_layer_heading_probe(store, title, int(unit_idx), cfg, radius_units=int(getattr(cfg.align, "pdf_text_probe_radius_units", 3)))
                    if isinstance(pdf_text_probe, dict) and bool(pdf_text_probe.get("hit")):
                        unit_idx = int(pdf_text_probe.get("unit", unit_idx))
                        b_fallbacks.append(f"pdf_text_probe_ch{no}")

                if need_vl:
                    retries = 1
                    if bool(getattr(cfg.align, "vl_boundary_relocate_retry_on_weak", False)):
                        retries = max(1, int(getattr(cfg.align, "vl_boundary_relocate_max_retries", 2) or 2))
                    base_radius = int(getattr(cfg.align, "vl_boundary_relocate_radius_units", 4) or 4)
                    for attempt in range(retries):
                        if attempt > 0:
                            # widen search + raise DPI on retry
                            setattr(cfg.pdf, "unit_image_dpi_low", int(getattr(cfg.pdf, "unit_image_dpi_high", 320)))
                            setattr(cfg.align, "vl_boundary_relocate_radius_units", base_radius + attempt)
                        vl_unit_reloc = _vl_relocate_unit_for_chapter_start(
                            store,
                            session,
                            cfg,
                            chapter_no=no,
                            chapter_title=title,
                            base_unit=int(unit_idx),
                            min_unit=int(prev_u_bound),
                            max_unit=int(next_u_bound),
                            logger=logger,
                        )
                        # restore radius if modified
                        setattr(cfg.align, "vl_boundary_relocate_radius_units", base_radius)
                        if isinstance(vl_unit_reloc, dict) and bool(vl_unit_reloc.get("applied")):
                            unit_idx = int(vl_unit_reloc.get("best_unit", unit_idx))
                            b_fallbacks.append(f"vl_boundary_relocate_ch{no}")
                            suspect_reasons.append(f"vl_unit_relocated:{int(vl_unit_reloc.get('base_unit', unit_idx))}->{unit_idx}")
                            break
                        score_try = float((vl_unit_reloc or {}).get("best_score", 0.0) or 0.0) if isinstance(vl_unit_reloc, dict) else 0.0
                        if score_try >= float(getattr(cfg.align, "vl_boundary_verify_min_score", 0.78)):
                            break
            except Exception as ex:
                try:
                    logger.warning("[BOUNDARY][VL] ch=%s relocate failed: %s", no, ex)
                except Exception:
                    pass

            est = _estimate_raw_from_unit(unit_idx, unit_count=unit_count, raw_len=raw_len, points=interp_points)
            lo = max(prev + 1, est - fb_window // 2)
            hi = min(raw_len, est + fb_window // 2)
            if hi <= lo:
                lo = max(prev + 1, 0)
                hi = min(raw_len, lo + fb_window)

            # If we do not have a usable opening snippet, skip LLM fallback (it becomes noise-driven).
            min_opening = int(getattr(cfg.align, "opening_min_chars_for_llm_fallback", 40) or 40)
            if not opening or len(str(opening).strip()) < min_opening:
                # Deterministic fallback: trust the unit->raw estimate and enforce monotonicity.
                final = int(est)
                final = max(prev + max(1, min_seg_chars), final)
                final = min(raw_len - 1, final) if raw_len > 0 else 0
                boundary_details.append({
                    "chapter_no": no,
                    "title": title,
                    "unit_start": unit_idx,
                    "opening_len": len(str(opening or "")),
                    "method": "fallback_est_no_opening",
                    "lo": int(lo),
                    "hi": int(hi),
                    "est": int(est),
                    "picked_offset": None,
                    "final_start_raw": int(final),
                    "confidence": 0.0,
                    "notes": "opening_snippet too short/empty; skipped LLM boundary pick",
                    "candidates_init": candidates_init[:5],
                    "candidates_init_full": candidates_init,
                    "candidate_policy_flags": candidate_policy_flags,
                    "suspect_reasons": suspect_reasons,
                    "vl_unit_relocate": vl_unit_reloc,
                })
                issues.append({
                    "kind": "boundary_fallback_skipped_llm",
                    "chapter_no": no,
                    "severity": "medium",
                    "notes": f"opening_len={len(str(opening or ''))} < min_opening={min_opening}; used est={int(est)}",
                })
                final_bounds.append(int(final))
                prev = int(final)
                continue

            # LLM fallback can return null when the window doesn't include the true start.
            # Retry with an expanded window a small number of times.
            llm_retries = int(getattr(cfg.align, 'llm_fallback_retries', 0) or 0)
            expand_factor = float(getattr(cfg.align, 'llm_fallback_expand_factor', 1.6) or 1.6)
            use_raw_for_llm = bool(getattr(cfg.align, 'llm_use_raw_excerpt_for_tex', True))
            structural_fallback = bool(getattr(cfg.align, 'structural_heading_fallback_enable', True))

            picked = None
            final = None
            notes = ""
            conf = 0.0
            anchor_snip = ""
            cur_lo, cur_hi = int(lo), int(hi)

            for attempt in range(max(0, llm_retries) + 1):
                stride_candidates = _build_candidates_by_stride(
                    match_text, cur_lo, cur_hi, opening,
                    stride=cand_stride,
                    win=cand_win,
                    topk=cand_topk,
                    candidate_min_alnum_ratio=float(getattr(cfg.align, 'candidate_min_alnum_ratio', 0.06) or 0.06),
                    candidate_max_brace_ratio=float(getattr(cfg.align, 'candidate_max_brace_ratio', 0.22) or 0.22),
                )

                # also include initial candidates as explicit options (even if low conf)
                if chosen is not None:
                    off0 = max(cur_lo, min(cur_hi - 1, int(chosen)))
                    ex0 = match_text[off0:min(raw_len, off0 + cand_win)]
                    sc0 = _local_snippet_score(opening, ex0)
                    stride_candidates = [(off0, ex0, sc0)] + [c for c in stride_candidates if abs(c[0] - off0) > cand_win // 3]
                    stride_candidates = stride_candidates[:cand_topk]

                # For TeX inputs, show raw (non-shadow) excerpts to the LLM; it is more informative than blanks.
                if bool(is_tex) and use_raw_for_llm:
                    llm_candidates = [(off, raw_text[off:min(raw_len, off + cand_win)], sc) for off, _ex, sc in stride_candidates]
                else:
                    llm_candidates = stride_candidates

                picked = _llm_pick_boundary(
                    session, cfg,
                    chapter_no=no,
                    chapter_title=title,
                    opening_snippet=opening,
                    anchors=anchors,
                    candidates=llm_candidates,
                )

                llm_pick_failed = False
                if isinstance(picked, dict) and (picked.get("_parse_error") or picked.get("_exception")):
                    llm_pick_failed = True
                    err_tag = "parse" if picked.get("_parse_error") else "exception"
                    b_fallbacks.append(f"llm_pick_failed_ch{no}:{err_tag}")
                    notes = f"llm_pick_failed:{err_tag}"
                    conf = 0.0
                    anchor_snip = ""
                    best_off = None
                else:
                    best_off = picked.get("best_offset") if isinstance(picked, dict) else None
                    conf = float(picked.get("confidence", 0.0) or 0.0) if isinstance(picked, dict) else 0.0
                    anchor_snip = str(picked.get("anchor_snippet") or "").strip() if isinstance(picked, dict) else ""
                    notes = str(picked.get("notes") or "").strip() if isinstance(picked, dict) else ""

                # Normalize best_offset (LLM may return absolute or relative-to-window offsets)
                best_off_raw = best_off
                best_off_abs = None
                best_off_was_relative = False
                if isinstance(best_off, int):
                    if int(cur_lo) <= best_off <= int(cur_hi):
                        best_off_abs = int(best_off)
                    elif 0 <= best_off <= (int(cur_hi) - int(cur_lo)):
                        best_off_abs = int(cur_lo) + int(best_off)
                        best_off_was_relative = True
                    elif (int(cur_lo) - 200) <= best_off <= (int(cur_hi) + 200):
                        best_off_abs = max(int(cur_lo), min(int(cur_hi), int(best_off)))

                # Choose best_offset first (structural headings should NOT be overridden by snippets).
                final = None
                if isinstance(best_off_abs, int):
                    final = int(best_off_abs)
                    if best_off_was_relative:
                        notes = (notes + " | " if notes else "") + "best_offset interpreted as relative"

                # If best_offset is missing/unusable, try to locate a unique anchor_snippet in-window.
                # Use robust matching (normalized + fuzzy) but require uniqueness.
                if final is None and anchor_snip:
                    base = raw_text if (bool(is_tex) and use_raw_for_llm) else match_text

                    def _find_unique_anchor_pos_abs(base_text: str, lo: int, hi: int, snip: str) -> Tuple[Optional[int], str]:
                        sub = base_text[lo:hi]
                        # exact
                        try:
                            cnt = sub.count(snip)
                        except Exception:
                            cnt = 0
                        if cnt == 1:
                            p = sub.find(snip)
                            if p >= 0:
                                return lo + int(p), "exact_unique"
                            return None, "exact_find_failed"
                        if cnt > 1:
                            return None, f"exact_nonunique:{cnt}"

                        # normalized exact
                        try:
                            nsub, nmap = normalize_text_for_match(sub)
                            nsnip, _ = normalize_text_for_match(snip)
                            if nsnip:
                                ncnt = nsub.count(nsnip)
                                if ncnt == 1:
                                    npos = int(nsub.find(nsnip))
                                    raw_rel = int(nmap[npos]) if 0 <= npos < len(nmap) else None
                                    if raw_rel is not None:
                                        return lo + raw_rel, "norm_unique"
                                if ncnt > 1:
                                    return None, f"norm_nonunique:{ncnt}"
                        except Exception:
                            pass

                        # fuzzy (clearly dominant)
                        try:
                            from rapidfuzz import fuzz as rf_fuzz
                            nsub, nmap = normalize_text_for_match(sub)
                            nsnip, _ = normalize_text_for_match(snip)
                        except Exception:
                            return None, "fuzzy_unavailable"
                        if not nsnip or not nsub or len(nsnip) < 20:
                            return None, "too_short"
                        key = nsnip[: min(40, len(nsnip))]
                        hits: List[int] = []
                        pos0 = 0
                        while True:
                            j = nsub.find(key, pos0)
                            if j < 0:
                                break
                            hits.append(int(j))
                            pos0 = j + 1
                            if len(hits) > 30:
                                break
                        if not hits:
                            return None, "no_key_hits"
                        best = (-1.0, None)
                        second = (-1.0, None)
                        for j in hits:
                            seg = nsub[j: min(len(nsub), j + len(nsnip) + 200)]
                            sc = float(rf_fuzz.partial_ratio(nsnip, seg) or 0.0)
                            if sc > best[0]:
                                second = best
                                best = (sc, j)
                            elif sc > second[0]:
                                second = (sc, j)
                        min_sc = float(getattr(cfg.align, 'opening_align_min_score', 55.0) or 55.0)
                        margin = float(getattr(cfg.align, 'opening_align_margin', 6.0) or 6.0)
                        if best[1] is not None and best[0] >= min_sc and (best[0] - second[0]) >= margin:
                            npos = int(best[1])
                            raw_rel = int(nmap[npos]) if 0 <= npos < len(nmap) else None
                            if raw_rel is not None:
                                return lo + raw_rel, f"fuzzy_unique:{best[0]:.1f}:{second[0]:.1f}"
                        return None, f"fuzzy_not_unique:{best[0]:.1f}:{second[0]:.1f}"

                    pos_abs, note = _find_unique_anchor_pos_abs(base, int(cur_lo), int(cur_hi), str(anchor_snip))
                    if pos_abs is not None:
                        final = int(pos_abs)
                        notes = (notes + " | " if notes else "") + f"anchor_snip_used:{note}"
                    else:
                        notes = (notes + " | " if notes else "") + f"anchor_snip_not_used:{note}"

                # Backward-anchor tail-cut: if the anchor snippet was taken *before* the nominal
                # start unit, boundaries tend to land on snippet start; shift to snippet end.
                backward_tail_applied = False
                backward_tail_score = None
                backward_tail_delta = None
                try:
                    anchor_unit = int(ch.get("anchor_unit") or -1)
                except Exception:
                    anchor_unit = -1
                is_backward_anchor = anchor_unit >= 0 and anchor_unit < int(ch.get("unit_start") or 0)
                if final is not None and is_backward_anchor and opening:
                    try:
                        from rapidfuzz import fuzz as rf_fuzz
                        scan_chars = int(getattr(cfg.align, "backward_anchor_scan_chars", 12000))
                        # rapidfuzz.partial_ratio_alignment.score is in [0..100].
                        # Allow config to be specified either as ratio (<=1) or score (0..100).
                        tail_min = float(getattr(cfg.align, "backward_anchor_tail_min_score", 60.0))
                        if tail_min <= 1.0:
                            tail_min = tail_min * 100.0
                        shift_fwd = int(getattr(cfg.align, "backward_anchor_shift_chars", 1200))
                        seg = match_text[int(final) : min(raw_len, int(final) + max(2000, scan_chars))]
                        align = rf_fuzz.partial_ratio_alignment(_norm_for_match(opening), _norm_for_match(seg))
                        if align and float(getattr(align, "score", 0.0)) >= tail_min:
                            dest_end = int(getattr(align, "dest_end", 0))
                            if dest_end > 10:
                                new_final = int(final) + dest_end
                                if new_final <= int(cur_hi):
                                    backward_tail_applied = True
                                    backward_tail_score = float(getattr(align, "score", 0.0))
                                    backward_tail_delta = int(new_final - int(final))
                                    final = new_final
                        if final is not None and (not backward_tail_applied) and shift_fwd > 0:
                            new_final = min(int(cur_hi), int(final) + shift_fwd)
                            backward_tail_delta = int(new_final - int(final))
                            final = new_final
                    except Exception:
                        pass

                if final is not None:
                    break

                # If the model returned null, expand window and retry.
                if attempt < max(0, llm_retries):
                    w = max(1, int(cur_hi) - int(cur_lo))
                    extra = int(max(2000, w * (max(1.1, expand_factor) - 1.0) / 2.0))
                    cur_lo = max(prev + 1, int(cur_lo) - extra)
                    cur_hi = min(raw_len, int(cur_hi) + extra)

            # Optional structural fallback: locate a chapter heading pattern inside the (possibly expanded) window.
            if final is None and structural_fallback:
                try:
                    base = raw_text if bool(is_tex) else match_text
                    win_txt = base[cur_lo:cur_hi]
                    pat = None
                    if bool(is_tex):
                        # Prefer explicit TeX headings if present.
                        for p in [r"\\chapter\s*\*?\s*\{", r"\\section\s*\*?\s*\{", r"\\begin\{chapter\}"]:
                            if re.search(p, win_txt):
                                pat = p
                                break
                    else:
                        for p in [r"(?im)^\s*chapter\s+\d+\b", r"(?im)^\s*CHAPTER\s+\d+\b"]:
                            if re.search(p, win_txt):
                                pat = p
                                break
                    if pat:
                        m = re.search(pat, win_txt)
                        if m:
                            final = cur_lo + int(m.start())
                            notes = (notes + f" | structural_heading_fallback={pat}").strip(" |")
                except Exception:
                    pass
            if final is None:
                # last-resort: keep monotone + estimated
                final = max(prev + 1, min(raw_len, est))
                # Avoid pathological collapses when estimate is non-informative
                if prev >= 0 and (final - prev) <= catastrophic_gap_chars:
                    final = min(raw_len, prev + max(1, min_chapter_chars))
                issues.append({
                    "chapter_no": no,
                    "type": "boundary_fallback_failed",
                    "severity": "high",
                    "notes": notes or "LLM returned null; using estimate",
                    "window": [int(cur_lo), int(cur_hi)],
                })
                b_fallbacks.append(f"fallback_failed_ch{no}")
            else:
                final = max(prev + 1, min(raw_len, int(final)))
                # Avoid pathological collapses (e.g., prev+1 chains)
                if prev >= 0 and (final - prev) <= catastrophic_gap_chars:
                    final = min(raw_len, prev + max(1, min_chapter_chars))
                if not (int(cur_lo) <= final <= int(cur_hi)):
                    issues.append({
                        "chapter_no": no,
                        "type": "boundary_out_of_window",
                        "severity": "medium",
                        "notes": notes,
                        "window": [int(cur_lo), int(cur_hi)],
                        "best_offset": int(final),
                    })
                    b_fallbacks.append(f"fallback_out_of_window_ch{no}")

            boundary_details.append({
                "chapter_no": no,
                "start_raw": int(final),
                "method": "llm_fallback",
                "conf": conf,
                "local_snippet_score": local,
                "toc_likeness": toc_like,
                "candidates_init": candidates_init[:5],
                "candidates_init_full": candidates_init,
                "candidate_policy_flags": candidate_policy_flags,
                "suspect_reasons": suspect_reasons,
                "vl_unit_relocate": vl_unit_reloc,
                "status": ("fallback_ok" if (conf >= interp_min_conf and (int(final) - prev) >= min_seg_chars) else "fallback_unreliable"),
                "notes": notes,
                "window": [int(cur_lo), int(cur_hi)],
                "best_offset_raw": best_off_raw if "best_off_raw" in locals() else best_off,
                "best_offset_abs": best_off_abs if "best_off_abs" in locals() else None,
                "best_offset_was_relative": bool(best_off_was_relative) if "best_off_was_relative" in locals() else False,
                "is_backward_anchor": bool(is_backward_anchor) if "is_backward_anchor" in locals() else False,
                "backward_tail_applied": bool(backward_tail_applied) if "backward_tail_applied" in locals() else False,
                "backward_tail_score": backward_tail_score if "backward_tail_score" in locals() else None,
                "backward_tail_delta": backward_tail_delta if "backward_tail_delta" in locals() else None,
                "llm_window_expanded": bool(int(cur_lo) != int(lo) or int(cur_hi) != int(hi)),
                "llm_attempts": int(attempt) + 1 if 'attempt' in locals() else 1,
                "picked": picked,
                "stride_candidates": [(int(o), float(s)) for o, _ex, s in (stride_candidates or [])][:min(10, len(stride_candidates or []))],
            })
            final_bounds.append(int(final))
            # Update interpolation anchors only when the boundary is reliable; avoid poisoning mapping with bad points.
            if conf >= interp_min_conf and (int(final) - prev) >= min_seg_chars:
                interp_points.append((unit_idx, int(final)))
            else:
                issues.append({
                    "chapter_no": no,
                    "type": "interp_anchor_skipped_unreliable",
                    "severity": "medium",
                    "notes": f"conf={conf:.2f}, seg_len={int(final)-prev} < min_seg={min_seg_chars} or conf<{interp_min_conf:.2f}",
                })
            prev = int(final)

        # final monotone sanity: ensure strictly increasing (except last may be raw_len)
        for i in range(1, len(final_bounds)):
            if final_bounds[i] <= final_bounds[i-1]:
                final_bounds[i] = min(raw_len, final_bounds[i-1] + 1)
                issues.append({
                    "chapter_no": int(chapters_anchored[i].get("no", i+1) or (i+1)),
                    "type": "monotone_repair",
                    "severity": "high",
                    "notes": "post-pass monotone repair applied",
                })

        # TeX chapter-heading lock (pre-snap): align each chapter cut to the nearest matched
        # TeX chapter heading when evidence is strong. This explicitly prevents recurrent
        # cross-chapter leakage around chapter titles.
        tex_heading_lock_report_pre = {"enabled": False, "items": []}
        if is_tex and bool(getattr(cfg.align, "tex_heading_lock_enable", True)):
            try:
                final_bounds, tex_heading_lock_report_pre = _apply_tex_heading_lock(
                    final_bounds,
                    boundary_details,
                    chapters_anchored,
                    tex_heading_events_for_lock,
                    cfg=cfg,
                )
            except Exception:
                tex_heading_lock_report_pre = {"enabled": False, "items": [], "error": "pre_lock_exception"}

        # Safety snap: avoid cutting inside TeX math / macros by snapping boundaries
        # to structural/newline positions. This does NOT rewrite正文; it only adjusts
        # the cut index.
        pre_snap_bounds = list(final_bounds)
        snap_window = int(getattr(cfg.align, "safe_cut_window_chars", 2000) or 2000)
        allow_back = int(getattr(cfg.align, "safe_cut_allow_backshift_chars", 600) or 600)
        min_positions = [max(0, int(x) - allow_back) for x in final_bounds]
        mid_pen = int(getattr(cfg.align, "safe_cut_midline_penalty", 250) or 250)
        ch_bonus = int(getattr(cfg.align, "safe_cut_strong_chapter_bonus", 800) or 800)
        generic_heading_bonus = int(getattr(cfg.align, "safe_cut_generic_heading_bonus", 30) or 30)
        tex_nonchapter_heading_bonus = int(getattr(cfg.align, "safe_cut_tex_nonchapter_heading_bonus", 10) or 10)
        tex_decimal_heading_penalty = int(getattr(cfg.align, "safe_cut_tex_decimal_heading_penalty", 140) or 140)
        snapped_bounds, snap_report_pre = snap_boundaries(
            raw_text,
            final_bounds,
            is_tex=bool(is_tex),
            window=snap_window,
            min_gap=int(getattr(cfg.align, 'post_min_gap_chars', 1) or 1),
            min_positions=min_positions,
            midline_penalty=mid_pen,
            strong_chapter_bonus=ch_bonus,
            generic_heading_bonus=generic_heading_bonus,
            tex_nonchapter_heading_bonus=tex_nonchapter_heading_bonus,
            tex_decimal_heading_penalty=tex_decimal_heading_penalty,
            avoid_midword=bool(getattr(cfg.align, 'safe_cut_avoid_midword', True)),
            midword_penalty=float(getattr(cfg.align, 'safe_cut_midword_penalty', 0.35) or 0.35),
            avoid_tex_command_split=bool(getattr(cfg.align, 'safe_cut_avoid_tex_command_split', True)),
            tex_command_penalty=float(getattr(cfg.align, 'safe_cut_tex_command_penalty', 0.8) or 0.8),
        )

        # Post-check: boundary collapse repair (min-gap enforcement on unreliable boundaries)
        repaired_bounds, repair_report = _post_repair_bounds_min_gap(snapped_bounds, boundary_details, raw_len=raw_len, cfg=cfg, logger=logger)
        if repaired_bounds != snapped_bounds:
            dump_json(cache_dir / "cut_snap_report_pre_repair.json", snap_report_pre)
            summary["artifacts"]["cut_snap_report_pre_repair"] = str(cache_dir / "cut_snap_report_pre_repair.json")

            min_positions2 = [max(0, int(x) - allow_back) for x in repaired_bounds]
            snapped_bounds2, snap_report_post = snap_boundaries(
                raw_text,
                repaired_bounds,
                is_tex=bool(is_tex),
                window=snap_window,
                min_gap=int(getattr(cfg.align, 'post_min_gap_chars', 1) or 1),
                min_positions=min_positions2,
                midline_penalty=mid_pen,
                strong_chapter_bonus=ch_bonus,
                generic_heading_bonus=generic_heading_bonus,
                tex_nonchapter_heading_bonus=tex_nonchapter_heading_bonus,
                tex_decimal_heading_penalty=tex_decimal_heading_penalty,
                avoid_midword=bool(getattr(cfg.align, 'safe_cut_avoid_midword', True)),
                midword_penalty=float(getattr(cfg.align, 'safe_cut_midword_penalty', 0.35) or 0.35),
            )
            final_bounds = snapped_bounds2

            # update boundary_details to reflect final cut points
            bd_updates = []
            for ii in range(min(len(boundary_details), len(final_bounds))):
                before = boundary_details[ii].get("start_raw")
                boundary_details[ii]["start_raw"] = int(final_bounds[ii])
                if before != int(final_bounds[ii]):
                    bd_updates.append({"i": ii, "before": int(before) if before is not None else None, "after": int(final_bounds[ii])})
            repair_report["boundary_details_updates"] = bd_updates
            repair_report["snap_report_pre"] = snap_report_pre
            repair_report["snap_report_post"] = snap_report_post
            dump_json(cache_dir / "boundary_repair_report.json", repair_report)
            summary["artifacts"]["boundary_repair_report"] = str(cache_dir / "boundary_repair_report.json")

            dump_json(cache_dir / "cut_snap_report.json", snap_report_post)
            summary["artifacts"]["cut_snap_report"] = str(cache_dir / "cut_snap_report.json")
            snap_report = snap_report_post
        else:
            final_bounds = snapped_bounds
            # keep boundary_details consistent
            for ii in range(min(len(boundary_details), len(final_bounds))):
                boundary_details[ii]["start_raw"] = int(final_bounds[ii])
            dump_json(cache_dir / "cut_snap_report.json", snap_report_pre)
            summary["artifacts"]["cut_snap_report"] = str(cache_dir / "cut_snap_report.json")
            snap_report = snap_report_pre

        # TeX chapter-heading lock (post-snap): safe_cut can still drift to a nearby line. Re-apply the
        # chapter-level lock so the final cut remains anchored on the matched chapter heading.
        tex_heading_lock_report_post = {"enabled": False, "items": []}
        if is_tex and bool(getattr(cfg.align, "tex_heading_lock_enable", True)):
            try:
                final_bounds, tex_heading_lock_report_post = _apply_tex_heading_lock(
                    final_bounds,
                    boundary_details,
                    chapters_anchored,
                    tex_heading_events_for_lock,
                    cfg=cfg,
                )
                # Keep boundary_details consistent with final bounds after the post-lock.
                for ii in range(min(len(boundary_details), len(final_bounds))):
                    boundary_details[ii]["start_raw"] = int(final_bounds[ii])
            except Exception:
                tex_heading_lock_report_post = {"enabled": False, "items": [], "error": "post_lock_exception"}
        # Final safe-cut snap after post-lock to avoid command-prefix splits (\ + chapter).
        try:
            if bool(getattr(cfg.align, "safe_cut_after_post_lock", True)):
                _min_positions = [max(0, int(x) - allow_back) for x in final_bounds]
                final_bounds, snap_report_final = snap_boundaries(
                    raw_text,
                    final_bounds,
                    is_tex=bool(is_tex),
                    window=snap_window,
                    min_gap=int(getattr(cfg.align, 'post_min_gap_chars', 1) or 1),
                    min_positions=_min_positions,
                    midline_penalty=mid_pen,
                    strong_chapter_bonus=ch_bonus,
                    generic_heading_bonus=generic_heading_bonus,
                    tex_nonchapter_heading_bonus=tex_nonchapter_heading_bonus,
                    tex_decimal_heading_penalty=tex_decimal_heading_penalty,
                    avoid_midword=bool(getattr(cfg.align, 'safe_cut_avoid_midword', True)),
                    midword_penalty=float(getattr(cfg.align, 'safe_cut_midword_penalty', 0.35) or 0.35),
                    avoid_tex_command_split=bool(getattr(cfg.align, 'safe_cut_avoid_tex_command_split', True)),
                    tex_command_penalty=float(getattr(cfg.align, 'safe_cut_tex_command_penalty', 0.8) or 0.8),
                )
                for ii in range(min(len(boundary_details), len(final_bounds))):
                    boundary_details[ii]["start_raw"] = int(final_bounds[ii])
            else:
                snap_report_final = {"enabled": False, "items": []}
        except Exception:
            snap_report_final = {"enabled": False, "items": [], "error": "final_snap_exception"}

        dump_json(cache_dir / "tex_heading_lock_report.json", {
            "pre": tex_heading_lock_report_pre,
            "post": tex_heading_lock_report_post,
            "final_snap": snap_report_final,
        })
        summary["artifacts"]["tex_heading_lock_report"] = str(cache_dir / "tex_heading_lock_report.json")

        # Boundary decision trace (lightweight, per-chapter):
        # - Makes "strategies" observable (no silent empty branches)
        # - Enables debugging from artifacts alone without re-running
        try:
            snap_items = (snap_report or {}).get("items", []) if isinstance(snap_report, dict) else []
            snap_by_i = {int(it.get("i")): it for it in snap_items if isinstance(it, dict) and it.get("i") is not None}
        except Exception:
            snap_by_i = {}

        run_id_env = os.environ.get("PDFTXTALIGN_RUN_ID")
        code_version_env = os.environ.get("PDFTXTALIGN_CODE_VERSION")
        code_hash_env = os.environ.get("PDFTXTALIGN_CODE_HASH")
        cfg_hash_env = os.environ.get("PDFTXTALIGN_CONFIG_HASH")

        chapter_decisions = []
        for i, ch in enumerate(chapters_anchored):
            bd = boundary_details[i] if i < len(boundary_details) else {}
            pre_b = pre_snap_bounds[i] if i < len(pre_snap_bounds) else None
            post_b = final_bounds[i] if i < len(final_bounds) else None
            snap = snap_by_i.get(i, {})

            # Promote boundary metadata back to chapter entries for naming/reporting.
            try:
                if post_b is not None:
                    ch["text_start"] = int(post_b)
                # matched headings
                if bd.get("matched_tex_heading") or bd.get("tex_heading_lock_title"):
                    ch["matched_tex_heading"] = str(bd.get("matched_tex_heading") or bd.get("tex_heading_lock_title") or "")
                if isinstance(bd.get("vl_relocate"), dict):
                    _vh = str((bd.get("vl_relocate") or {}).get("best_title") or "").strip()
                    if _vh:
                        ch["matched_pdf_heading"] = _vh
                if bd.get("reclaim_next_heading"):
                    ch["boundary_repair_actions"] = list(dict.fromkeys((ch.get("boundary_repair_actions") or []) + ["reclaim_next_heading"]))
                # start source / confidence
                if bd.get("reclaim_next_heading"):
                    ch["start_source"] = "tex_lock_reclaim"
                elif bd.get("tex_heading_lock_applied"):
                    ch["start_source"] = "tex_lock"
                elif any((f"vl_boundary_relocate_ch{ch.get('no')}" == str(f)) for f in (b_fallbacks or [])):
                    ch["start_source"] = "vl_relocate"
                elif any((f"pdf_text_probe_ch{ch.get('no')}" == str(f)) for f in (b_fallbacks or [])):
                    ch["start_source"] = "pdf_text_probe"
                else:
                    ch["start_source"] = str(bd.get("method") or "fallback")
                conf_cands = [bd.get("conf"), bd.get("local_snippet_score"), bd.get("lead_score"), ((bd.get("vl_relocate") or {}).get("best_score") if isinstance(bd.get("vl_relocate"), dict) else None)]
                for _c in conf_cands:
                    if _c is not None:
                        try:
                            ch["start_confidence"] = float(_c)
                            break
                        except Exception:
                            pass
                ch["expected_title"] = str(ch.get("title_corrected") or ch.get("title") or "")
                ch["is_low_confidence"] = bool(float(ch.get("start_confidence", 0.0) or 0.0) < float(getattr(cfg.align, "vl_boundary_verify_min_score", 0.78)))
            except Exception:
                pass

            flags = []
            try:
                if bd.get("best_offset_was_relative"):
                    flags.append("llm_best_offset_relative")
                if bd.get("is_backward_anchor"):
                    flags.append("backward_anchor")
                if bd.get("backward_tail_applied"):
                    flags.append("backward_anchor_tail_cut")
                if bd.get("llm_window_expanded"):
                    flags.append("llm_window_expanded")
                if isinstance(str(bd.get("notes") or ""), str) and "structural_heading_fallback" in str(bd.get("notes") or ""):
                    flags.append("structural_heading_fallback")
                if bd.get("tex_heading_lock_applied"):
                    flags.append("tex_heading_lock")
            except Exception:
                pass
            try:
                d = int(snap.get("delta", 0) or 0)
                if d != 0:
                    flags.append("safe_cut_snapped")
                rule = str(snap.get("rule") or "")
                if "+min_pos" in rule:
                    flags.append("safe_cut_min_pos")
                if "+monotone" in rule:
                    flags.append("safe_cut_monotone")
            except Exception:
                pass

            chapter_decisions.append({
                "i": int(i),
                "no": ch.get("no"),
                "title": ch.get("title"),
                "printed_page": ch.get("printed_page"),
                "unit_start": ch.get("unit_start"),
                "anchor_unit": ch.get("anchor_unit"),
                "pre_snap_boundary": int(pre_b) if pre_b is not None else None,
                "final_boundary": int(post_b) if post_b is not None else None,
                "snap_delta": int(snap.get("delta")) if isinstance(snap, dict) and snap.get("delta") is not None else None,
                "snap_rule": str(snap.get("rule")) if isinstance(snap, dict) else None,
                "snap_heading": str(snap.get("heading")) if isinstance(snap, dict) and snap.get("heading") is not None else None,
                "snap_candidates": (snap.get("candidates") or [])[:20] if isinstance(snap, dict) and isinstance(snap.get("candidates"), list) else [],
                "safe_cut_candidates": {
                    "n": int(len(snap.get("candidates") or [])) if isinstance(snap, dict) and isinstance(snap.get("candidates"), list) else 0,
                    "has_strong_chapter_heading": bool(snap.get("has_strong_chapter_heading")) if isinstance(snap, dict) else False,
                },
                "method": bd.get("method"),
                "conf": bd.get("conf"),
                "local_snippet_score": bd.get("local_snippet_score"),
                "lead_offset": bd.get("lead_offset"),
                "lead_score": bd.get("lead_score"),
                "lead_applied": bd.get("lead_applied"),
                "tex_heading_lock_applied": bd.get("tex_heading_lock_applied"),
                "tex_heading_lock_reason": bd.get("tex_heading_lock_reason"),
                "tex_heading_lock_hit": bd.get("tex_heading_lock_hit"),
                "start_source": ch.get("start_source"),
                "start_confidence": ch.get("start_confidence"),
                "expected_title": ch.get("expected_title"),
                "matched_tex_heading": ch.get("matched_tex_heading"),
                "matched_pdf_heading": ch.get("matched_pdf_heading"),
                "boundary_repair_actions": ch.get("boundary_repair_actions") or [],
                "is_low_confidence": ch.get("is_low_confidence"),
                "lead_offset_before": bd.get("lead_offset_before"),
                "lead_score_before": bd.get("lead_score_before"),
                "toc_likeness": bd.get("toc_likeness"),
                "candidates_init": bd.get("candidates_init"),
                "candidates_init_full": bd.get("candidates_init_full"),
                "candidate_policy_flags": bd.get("candidate_policy_flags"),
                "status": bd.get("status"),
                "suspect_reasons": bd.get("suspect_reasons"),
                "fallback_flags": [f for f in (b_fallbacks or []) if f"ch{ch.get('no')}" in str(f)],
                "window": bd.get("window"),
                "best_offset_abs": bd.get("best_offset_abs"),
                "best_offset_was_relative": bd.get("best_offset_was_relative"),
                "is_backward_anchor": bd.get("is_backward_anchor"),
                "backward_tail_applied": bd.get("backward_tail_applied"),
                "backward_tail_score": bd.get("backward_tail_score"),
                "backward_tail_delta": bd.get("backward_tail_delta"),
                "llm_attempts": bd.get("llm_attempts"),
                "llm_window_expanded": bd.get("llm_window_expanded"),
                "stride_candidates": bd.get("stride_candidates"),
                "notes": bd.get("notes"),
                "flags": flags,
            })

        dump_json(cache_dir / "boundary_decisions.json", {
            "schema": "boundary_decisions.v3",
            "run_id": str(run_id_env) if run_id_env else None,
            "code_version": str(code_version_env) if code_version_env else None,
            "code_hash": str(code_hash_env) if code_hash_env else None,
            "config_hash": str(cfg_hash_env) if cfg_hash_env else None,
            "config": {
                "lead_max_offset": int(getattr(cfg.align, "lead_max_offset", 600) or 600),
                "lead_scan_chars": int(getattr(cfg.align, "lead_scan_chars", 20000) or 20000),
                "lead_offset_strategy": str(getattr(cfg.align, "lead_offset_strategy", "shift") or "shift"),
                "safe_cut_midline_penalty": int(getattr(cfg.align, "safe_cut_midline_penalty", 250) or 250),
                "safe_cut_strong_chapter_bonus": int(getattr(cfg.align, "safe_cut_strong_chapter_bonus", 800) or 800),
                "safe_cut_tex_decimal_heading_penalty": int(getattr(cfg.align, "safe_cut_tex_decimal_heading_penalty", 140) or 140),
                "opening_align_min_score": float(getattr(cfg.align, "opening_align_min_score", 55.0) or 55.0),
                "backward_anchor_heading_snap_limit": int(getattr(cfg.align, "backward_anchor_heading_snap_limit", 40000) or 40000),
            },
            "raw_len": raw_len,
            "pre_snap_bounds": pre_snap_bounds,
            "final_bounds": final_bounds,
            "snap_report": snap_report,
            "fallbacks": list(b_fallbacks),
            "chapters": chapter_decisions,
        })
        summary["artifacts"]["boundary_decisions"] = str(cache_dir / "boundary_decisions.json")

        dump_json(cache_dir / "text_boundaries_final.json", {
            "boundaries": final_bounds,
            "details": boundary_details,
            "fallbacks": b_fallbacks,
        })
        summary["artifacts"]["text_boundaries"] = str(cache_dir / "text_boundaries_final.json")
        fallbacks.extend(b_fallbacks)

        # Verification (can be auto-skipped if unit_start mapping is missing, to avoid misleading low-hit flags)
        verify_report: Dict[str, Any] = {"enabled": False, "reason": "disabled_in_config"}
        fbv: List[str] = []
        if bool(getattr(cfg.verify, "enable", True)):
            try:
                missing_units = 0
                for _ch in chapters_mapped:
                    if _ch.get("unit_start") is None:
                        missing_units += 1
                miss_ratio = float(missing_units) / float(max(1, len(chapters_mapped)))
                skip_ratio = float(getattr(cfg.verify, "skip_if_missing_units_ratio", 0.50) or 0.50)
                if miss_ratio >= skip_ratio:
                    verify_report = {
                        "enabled": False,
                        "reason": "missing_unit_mapping",
                        "missing_units": int(missing_units),
                        "total_chapters": int(len(chapters_mapped)),
                        "missing_ratio": float(miss_ratio),
                    }
                    fbv.append("verify_skipped_missing_unit_mapping")
                    dump_json(cache_dir / "verify_report.json", verify_report)
                else:
                    verify_report, fbv = verify_chapter_segments(
                        store, session, cfg, chapters_anchored, raw_text, final_bounds, cache_dir, logger, match_text=match_text
                    )
            except Exception:
                # if verify crashes, do not block outputs; emit a minimal report
                verify_report = {"enabled": False, "reason": "verify_exception"}
                fbv.append("verify_exception")
                dump_json(cache_dir / "verify_report.json", verify_report)

        fallbacks.extend(fbv)
        summary["artifacts"]["verify_report"] = str(cache_dir / "verify_report.json")

        # Add verify-based review flags (does not block outputs)
        min_hit = float(getattr(cfg.verify, "verify_min_hit_ratio", 0.60))
        needs_thr = float(getattr(cfg.verify, "needs_review_hit_ratio", 0.35) or 0.35)
        min_hits = int(getattr(cfg.verify, "min_hits", 6) or 6)

        for chrep in (verify_report.get("chapters", []) if isinstance(verify_report, dict) else []):
            try:
                reason = str(chrep.get("reason") or "")
                if reason == "invalid_unit_range":
                    issues.append({
                        "chapter_no": chrep.get("no"),
                        "type": "verify_invalid_unit_mapping",
                        "severity": "medium",
                        "notes": "unit_start/unit_end missing or invalid; verify cannot run for this chapter",
                    })
                    continue
                if reason == "insufficient_samples":
                    issues.append({
                        "chapter_no": chrep.get("no"),
                        "type": "verify_insufficient_samples",
                        "severity": "low",
                        "notes": f"hits={int(chrep.get('hits', 0) or 0)} < min_hits={min_hits}",
                    })
                    continue

                hr = float(chrep.get("hit_ratio", 0.0) or 0.0)
                hits = int(chrep.get("hits", 0) or 0)
                ok = bool(chrep.get("ok", True))

                # Only interpret low-hit when verify actually ran on enough samples.
                if (not ok) and (hr < needs_thr):
                    issues.append({
                        "chapter_no": chrep.get("no"),
                        "type": "verify_low_hit_ratio",
                        "severity": "medium",
                        "notes": f"ok={ok}, hit_ratio={hr:.2f}, hits={hits}, needs_thr={needs_thr:.2f}",
                    })
                elif hr < min_hit:
                    issues.append({
                        "chapter_no": chrep.get("no"),
                        "type": "verify_low_hit_ratio",
                        "severity": "low",
                        "notes": f"hit_ratio={hr:.2f} < {min_hit:.2f}",
                    })
            except Exception:
                continue


        # Output writing
        try:
            if str(cfg.output.mode).lower() == "mark":
                out_path = book_out / f"{book_id}_marked.txt"
                _write_marked(raw_text, final_bounds, chapters_anchored, cfg, out_path)
                summary["artifacts"]["output"] = str(out_path)
            else:
                out_dir = book_out / "chapters"
                _write_split(raw_text, final_bounds, chapters_anchored, cfg, out_dir)
                summary["artifacts"]["output"] = str(out_dir)
        except Exception as ex:
            # Do not abort the whole pipeline on Windows path/IO issues; record and continue
            issues.append({
                "kind": "write_output_failed",
                "message": str(ex),
            })
            try:
                logger.warning("[OUTPUT] write failed: %s", ex)
            except Exception:
                pass
            summary["artifacts"]["output_error"] = str(ex)


        # Best-effort: derive number of chapter outputs for batch metrics.
        try:
            if summary.get("artifacts", {}).get("output"):
                outp = Path(str(summary["artifacts"]["output"]))
                if outp.exists() and outp.is_dir() and outp.name == "chapters":
                    summary["num_chapters"] = sum(1 for p in outp.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
                elif (cache_dir / "text_boundaries_final.json").exists():
                    tb = load_json(cache_dir / "text_boundaries_final.json", default={})
                    bds = (tb.get("boundaries", []) if isinstance(tb, dict) else [])
                    if isinstance(bds, list):
                        summary["num_chapters"] = int(len(bds))
        except Exception:
            pass

        # needs_review (A: do not pollute正文)
        _write_needs_review(book_out, book_id, issues, summary["artifacts"])

        # Fallback report for problematic books
        summary["fallbacks"] = list(dict.fromkeys(fallbacks))  # dedup preserve order
        if summary["fallbacks"]:
            fb_path = book_out / "fallback_report.md"
            fb_path.write_text(
                "# Fallback / Warning Report\n\n"
                f"- Book: {book_id}\n"
                f"- PDF: {pdf_path}\n"
                f"- TEXT: {text_path}\n\n"
                "## Flags\n" + "\n".join([f"- {x}" for x in summary["fallbacks"]]) + "\n\n"
                "## Key Artifacts\n" + "\n".join([f"- {k}: {v}" for k,v in summary["artifacts"].items()]) + "\n",
                encoding="utf-8"
            )
            summary["artifacts"]["fallback_report"] = str(fb_path)

        summary["success"] = True
        try:
            store.close()
        except Exception:
            pass
        # Persist reusable artifacts and delete large intermediates by default
        _persist_and_cleanup_cache(book_out, cache_dir, cfg, summary, logger)
        logger.info("=== DONE (success) ===")
        dump_json(book_out / "summary.json", summary)
        return summary

    except Exception as e:
        logger.error("=== FAILED ===")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        summary["success"] = False
        # Preserve already-collected fallback flags for auditing.
        summary["fallbacks"] = list(dict.fromkeys((summary.get("fallbacks") or []) + fallbacks + ["exception"]))
        # Ensure PDF handles are released on Windows even on failures.
        try:
            if store is not None:
                store.close()
        except Exception:
            pass
        # Persist reusable artifacts and delete large intermediates by default
        _persist_and_cleanup_cache(book_out, cache_dir, cfg, summary, logger)
        # write failure report
        fail = book_out / "failure_report.md"
        fail.write_text(
            "# Failure Report\n\n"
            f"- Book: {book_id}\n"
            f"- Error: {e}\n\n"
            "See logs:\n"
            f"- {log_path}\n",
            encoding="utf-8"
        )
        summary["artifacts"]["failure_report"] = str(fail)

        # Always emit a needs_review record for failures to keep the audit loop closed.
        try:
            issues = [{
                "code": "exception",
                "message": str(e),
                "trace": traceback.format_exc(),
                "fallbacks": list(summary.get("fallbacks") or []),
                "artifacts": dict(summary.get("artifacts") or {}),
            }]
            _write_needs_review(book_out, book_id, issues, summary.get("artifacts") or {})
        except Exception:
            pass
        dump_json(book_out / "summary.json", summary)
        return summary
