from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pathlib import Path
import logging
import re

from rapidfuzz.fuzz import partial_ratio

from .pdf_units import PDFUnitStore
from .llm_calls import vl_extract_short_phrases
from .text_processing import normalize_text_for_match as _normalize_text_for_match
from .utils import dump_json

def _norm(text: str) -> str:
    """Return normalized text string for matching (drops mapping)."""
    out = _normalize_text_for_match(text)
    # text_processing.normalize_text_for_match returns (norm_text, norm_to_raw_map)
    if isinstance(out, tuple) and out:
        return out[0]
    return out  # type: ignore[return-value]


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(1, union)


def _make_windows(text: str, window_chars: int) -> List[Tuple[str, str]]:
    """Return (name, window_text) windows to search."""
    t = text or ""
    if not t:
        return []
    w = max(2000, int(window_chars) if window_chars else 60000)
    n = len(t)

    windows: List[Tuple[str, str]] = []
    windows.append(("head", t[: min(n, w)]))

    if n > w:
        mid_c = n // 2
        lo = max(0, mid_c - w // 2)
        hi = min(n, mid_c + w // 2)
        windows.append(("mid", t[lo:hi]))

    if n > 2 * w:
        windows.append(("tail", t[max(0, n - w) : n]))

    # De-duplicate identical windows
    uniq: List[Tuple[str, str]] = []
    seen = set()
    for name, win in windows:
        key = (len(win), win[:256])
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, win))
    return uniq




def _snippet_align_diag(snippet: str, text: str) -> Tuple[float, Optional[int]]:
    """Return (score in [0,1], approx position) for snippet within text."""
    s = str(snippet or "").strip()
    t = str(text or "")
    if not s or not t:
        return 0.0, None
    try:
        score = partial_ratio(s.lower(), t.lower()) / 100.0
    except Exception:
        score = 0.0
    pos = None
    try:
        from rapidfuzz.fuzz import partial_ratio_alignment
        al = partial_ratio_alignment(s.lower()[:1800], t.lower()[:40000])
        if al is not None:
            pos = int(getattr(al, 'dest_start', -1) or -1)
            if pos < 0:
                pos = None
    except Exception:
        pos = None
    return float(score), pos


def _find_chapter_heading_offset_near_start(text: str, max_chars: int = 6000) -> Optional[int]:
    """Find the first strong chapter heading near the segment start (TeX or plain-text style)."""
    t = str(text or "")[: max(0, int(max_chars or 0))]
    if not t:
        return None
    # TeX heading commands first
    for m in re.finditer(r"(?m)^\s*(\\(?:chapter|section)\*?\{[^\n]{0,200}\})", t):
        line = m.group(1)
        if re.search(r"(?i)\bchapter\b", line) or re.search(r"第\s*[0-9０-９零〇○一二三四五六七八九十百千]+\s*章", line):
            return int(m.start(1))
    # Plain text headings
    for m in re.finditer(r"(?mi)^\s*(?:chapter\s*[0-9]{1,3}\b|第\s*[0-9０-９零〇○一二三四五六七八九十百千]+\s*章)", t):
        return int(m.start(0))
    return None

def verify_chapter_segments(
    store: PDFUnitStore,
    session_vl,
    cfg,
    chapter_plans: List[Dict[str, Any]],
    raw_text: str,
    boundaries: List[int],
    cache_dir: Path,
    logger: logging.Logger,
    match_text: str = None,
    segments: List[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[Dict[str, Any], List[str]]:
    """Closed-loop verification.

    For each chapter, sample a few PDF units and extract short phrases, then check whether
    those phrases can be found in the corresponding TXT/TeX segment.

    This verifier is intentionally conservative:
    - it searches within multiple windows (head/mid/tail) instead of the whole segment
    - it requires a minimum number of hits to avoid spurious matches
    - it uses both fuzzy score (partial_ratio) and token Jaccard as complementary signals
    """

    if not bool(getattr(cfg.verify, "enable", True)):
        return {"enabled": False}, ["verify_disabled"]

    samples_per = int(getattr(cfg.verify, "samples_per_chapter", 2) or 2)
    min_hit_ratio = float(getattr(cfg.verify, "verify_min_hit_ratio", 0.60) or 0.60)
    search_window_chars = int(getattr(cfg.verify, "search_window_chars", 60000) or 60000)

    fuzzy_th = float(getattr(cfg.verify, "fuzzy_threshold", 0.76) or 0.76)
    jac_th = float(getattr(cfg.verify, "token_jaccard_threshold", 0.42) or 0.42)
    min_hits = int(getattr(cfg.verify, "min_hits", 2) or 2)

    # Boundary-start diagnostics (uses PDF-derived opening snippets extracted earlier)
    start_check_enable = bool(getattr(cfg.verify, "start_boundary_check_enable", True))
    start_head_chars = int(getattr(cfg.verify, "start_boundary_head_chars", 20000) or 20000)
    prev_tail_chars = int(getattr(cfg.verify, "start_boundary_prev_tail_chars", 20000) or 20000)
    start_margin = float(getattr(cfg.verify, "start_boundary_prev_vs_head_margin", 0.08) or 0.08)
    start_min_opening = int(getattr(cfg.verify, "start_boundary_min_opening_chars", 24) or 24)
    start_prev_bad = float(getattr(cfg.verify, "start_boundary_prev_bad_score", 0.80) or 0.80)
    fail_on_prev_leak = bool(getattr(cfg.verify, "start_boundary_fail_on_prev_leak", True))
    heading_lead_warn = int(getattr(cfg.verify, "start_boundary_heading_lead_warn_chars", 2000) or 2000)

    report: Dict[str, Any] = {
        "enabled": True,
        "chapters": [],
        "summary": {},
    }

    vision_model = str(getattr(cfg.models, "vision_model", "qwen3-vl-plus"))
    enable_thinking = bool(getattr(cfg.models, "vision_enable_thinking", False))


    # Normalize/construct segments for verification.
    # Pipeline passes (chapter_plans, raw_text, boundaries, match_text). We build per-chapter segments with
    # unit ranges from chapter_plans and text slices from boundaries. We never rewrite正文.
    fallbacks: List[str] = []

    if segments is None:
        # boundaries are chapter start indices in raw_text
        if not isinstance(boundaries, list) or not boundaries:
            fallbacks.append("verify_missing_boundaries")
            segments = []
        else:
            ends = boundaries[1:] + [len(raw_text)]
            # unit ranges from chapter plans
            unit_starts: List[int] = []
            titles: List[str] = []
            nos: List[int] = []
            for j, ch in enumerate(chapter_plans or []):
                try:
                    nos.append(int(ch.get("no", j + 1) or (j + 1)))
                except Exception:
                    nos.append(j + 1)
                titles.append(str(ch.get("title_corrected") or ch.get("title") or ""))
                u = ch.get("unit_start")
                unit_starts.append(int(u) if u is not None else None)

            # If chapter_plans length differs from boundaries, align by min length
            n = min(len(boundaries), len(nos) if nos else len(boundaries))
            if n <= 0:
                fallbacks.append("verify_no_chapters")
                segments = []
            else:
                segments = []
                for j in range(n):
                    u0 = unit_starts[j] if j < len(unit_starts) else None
                    # unit_end uses next known unit_start or store.unit_count
                    u1 = None
                    if j + 1 < len(unit_starts) and unit_starts[j + 1] is not None:
                        u1 = int(unit_starts[j + 1])
                    elif hasattr(store, "unit_count"):
                        try:
                            u1 = int(store.unit_count)
                        except Exception:
                            u1 = None
                    seg_text = raw_text[boundaries[j]:ends[j]]
                    seg_match = None
                    if match_text is not None:
                        try:
                            seg_match = match_text[boundaries[j]:ends[j]]
                        except Exception:
                            seg_match = None
                    ch_plan = (chapter_plans[j] if isinstance(chapter_plans, list) and j < len(chapter_plans) else {}) or {}
                    segments.append({
                        "no": nos[j] if j < len(nos) else (j + 1),
                        "title": titles[j] if j < len(titles) else "",
                        "unit_start": u0,
                        "unit_end": u1,
                        "text": seg_text,
                        "match_text": seg_match,
                        "opening_snippet": str(ch_plan.get("opening_snippet") or ""),
                        "pdf_heading_hint": str(ch_plan.get("pdf_heading_hint") or ch_plan.get("title_corrected") or ch_plan.get("title") or ""),
                    })

    for seg_idx, seg in enumerate(segments):
        ch_no = int(seg.get("no", 0) or 0)
        title = str(seg.get("title", "") or "")
        u0 = seg.get("unit_start")
        u1 = seg.get("unit_end")
        text_seg = seg.get("text", "") or ""
        match_seg = seg.get("match_text") or text_seg

        start_diag = None
        start_leak_fail = False
        if start_check_enable:
            opening = str(seg.get("opening_snippet") or "")
            head_chunk = str(match_seg or "")[: max(0, start_head_chars)]
            prev_tail = ""
            if seg_idx > 0 and seg_idx - 1 < len(segments):
                prev_tail = str((segments[seg_idx - 1] or {}).get("match_text") or (segments[seg_idx - 1] or {}).get("text") or "")
                prev_tail = prev_tail[-max(0, prev_tail_chars):]
            head_score, head_pos = (0.0, None)
            prev_score, prev_pos = (0.0, None)
            if len(opening.strip()) >= start_min_opening:
                head_score, head_pos = _snippet_align_diag(opening, head_chunk)
                prev_score, prev_pos = _snippet_align_diag(opening, prev_tail)
            heading_offset = _find_chapter_heading_offset_near_start(text_seg, max_chars=max(start_head_chars, 6000))
            start_diag = {
                "opening_len": len(opening.strip()),
                "head_score": round(float(head_score), 4),
                "head_pos": int(head_pos) if head_pos is not None else None,
                "prev_tail_score": round(float(prev_score), 4),
                "prev_tail_pos": int(prev_pos) if prev_pos is not None else None,
                "pdf_heading_hint": str(seg.get("pdf_heading_hint") or ""),
                "detected_heading_offset": int(heading_offset) if heading_offset is not None else None,
            }
            # Boundary likely too late if PDF opening snippet matches previous segment tail much better than current head.
            if len(opening.strip()) >= start_min_opening and float(prev_score) >= start_prev_bad and float(prev_score) > float(head_score) + float(start_margin):
                start_diag["flag_prev_tail_stronger_than_head"] = True
                if fail_on_prev_leak:
                    start_leak_fail = True
            # Boundary likely too early/late alignment issue if a strong chapter heading appears far from segment start.
            if heading_offset is not None and int(heading_offset) > int(heading_lead_warn):
                start_diag["flag_heading_far_from_start"] = True

        if u0 is None or u1 is None or u0 >= u1:
            report["chapters"].append({
                "no": ch_no,
                "title": title,
                "ok": False,
                "reason": "invalid_unit_range",
            })
            continue

        # Choose sample units (deterministic): start + evenly spaced
        unit_list: List[int] = []
        span = max(1, int(u1) - int(u0))
        unit_list.append(int(u0))
        for i in range(1, samples_per):
            unit_list.append(int(u0) + int(round(i * span / max(1, samples_per))))
        unit_list = sorted(list({max(int(u0), min(int(u1 - 1), u)) for u in unit_list}))

        phrases: List[str] = []
        for ui in unit_list:
            img = store.render_unit(store.unit_ref(ui), dpi=store.dpi_low, region="body")
            out = vl_extract_short_phrases(
                session_vl,
                img,
                model=vision_model,
                enable_thinking=enable_thinking,
            )
            phs = (out or {}).get("phrases", [])
            if isinstance(phs, list):
                for p in phs:
                    s = str(p or "").strip()
                    # filter overly short/noisy phrases
                    if len(s) < 8:
                        continue
                    if sum(1 for ch in s if ch.isalnum()) < 6:
                        continue
                    phrases.append(s)

        # Deduplicate phrases
        uniq_phrases: List[str] = []
        seen = set()
        for p in phrases:
            key = _norm(p)[:80]
            if not key or key in seen:
                continue
            seen.add(key)
            uniq_phrases.append(p)
        phrases = uniq_phrases[: max(3, samples_per * 5)]

        # Prepare segment windows
        windows = _make_windows(match_seg, search_window_chars)
        norm_windows = [(name, _norm(w)) for name, w in windows]
        window_tokens = {name: _tokenize(norm) for name, norm in norm_windows}

        hits = 0
        details: List[Dict[str, Any]] = []

        for p in phrases:
            p_norm = _norm(p)
            p_toks = _tokenize(p_norm)

            best_score = 0.0
            best_where = None
            best_type = None

            # direct containment check first
            for name, w_norm in norm_windows:
                if p_norm and p_norm in w_norm:
                    best_score = 1.0
                    best_where = name
                    best_type = "contains"
                    break

            if best_type != "contains":
                for name, w_norm in norm_windows:
                    # fuzzy
                    fz = partial_ratio(p_norm, w_norm) / 100.0 if (p_norm and w_norm) else 0.0
                    # token-jaccard
                    jt = _jaccard(p_toks, window_tokens.get(name, [])) if len(p_toks) >= 4 else 0.0

                    # combine: take the stronger signal
                    score = max(fz, jt)
                    if score > best_score:
                        best_score = score
                        best_where = name
                        best_type = "fuzzy" if fz >= jt else "jaccard"

            ok = (best_type == "contains") or (best_score >= fuzzy_th) or (best_type == "jaccard" and best_score >= jac_th)
            if ok:
                hits += 1

            details.append({
                "phrase": p,
                "best_score": round(best_score, 4),
                "best_where": best_where,
                "best_type": best_type,
                "hit": bool(ok),
            })

        denom = max(1, len(phrases))
        hit_ratio = hits / denom
        ok = (hits >= min_hits) and (hit_ratio >= min_hit_ratio)
        if start_leak_fail:
            ok = False

        ch_report = {
            "no": ch_no,
            "title": title,
            "ok": bool(ok),
            "hits": hits,
            "phrases": len(phrases),
            "hit_ratio": round(hit_ratio, 4),
            "details": details[:50],
        }
        if start_diag is not None:
            ch_report["start_boundary"] = start_diag
        report["chapters"].append(ch_report)

    # Aggregate summary
    n = len(report["chapters"]) or 0
    ok_n = sum(1 for c in report["chapters"] if c.get("ok"))
    avg_hit = 0.0
    if n:
        avg_hit = sum(float(c.get("hit_ratio", 0.0) or 0.0) for c in report["chapters"]) / n

    report["summary"] = {
        "chapters": n,
        "ok_chapters": ok_n,
        "ok_ratio": round(ok_n / max(1, n), 4),
        "avg_hit_ratio": round(avg_hit, 4),
        "min_required_ok": min_hits,
        "min_hit_ratio": min_hit_ratio,
        "fuzzy_threshold": fuzzy_th,
        "token_jaccard_threshold": jac_th,
        "search_window_chars": search_window_chars,
    }

    dump_json(cache_dir / "verify_report.json", report)
    return report, fallbacks