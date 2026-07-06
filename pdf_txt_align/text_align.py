from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import re


TEXT_ALIGN_CACHE_VERSION = 5

from .text_processing import (
    parse_tex_chapters, parse_plain_chapters, build_tex_shadow_same_len,
    normalize_text_for_match, split_into_chunks
)
from .sequence_align import Chapter as Ch, align_chapter_sequences
from .similarity import ratio, partial_ratio, find_all, tokenize_simple, title_similarity
from .utils import dump_json, load_json, sanitize_filename, normalize_title
from .llm_calls import call_json
import bisect

@dataclass
class Boundary:
    chapter_no: int
    start_raw: int
    conf: float
    method: str

def _is_tex(text_path: Path) -> bool:
    return text_path.suffix.lower() in [".tex"]

def _llm_refine(session_llm, cfg, full_text: str, opening_snippet: str,
                candidates: List[Tuple[int, int, str]]) -> Optional[Dict[str, Any]]:
    """LLM refinement for chapter start.

    candidates: [(lo, hi, excerpt)] where [lo, hi) is the excerpt range in FULL_TEXT.
    """
    if not opening_snippet or not candidates:
        return None
    # limit payload
    max_cand = min(len(candidates), int(cfg.align.topk_candidates))
    candidates = candidates[:max_cand]
    prompt = (
        "You are aligning a PDF chapter opening snippet to the book text.\n"
        "Given the opening snippet (from the PDF) and several candidate excerpts from the TXT/TeX,\n"
        "choose the best candidate that contains the chapter start. Then output the exact character offset\n"
        "in the FULL text where the chapter begins (or closest possible). If unsure, output null.\n"
        "Return ONLY JSON: {best_offset:int|null, confidence:0..1, used_candidate:int|null, notes:string}\n"
        "Constraints:\n"
        "- The best_offset must be an absolute index into FULL_TEXT.\n"
        "- Prefer matching narrative text, not formulas.\n"
    )
    parts = [prompt, "\nOPENING_SNIPPET:\n", opening_snippet.strip(), "\n\nCANDIDATES:\n"]
    for i, (lo, hi, ex) in enumerate(candidates):
        parts.append(f"\n[Candidate {i}] range=[{lo},{hi})\n{ex}\n")
    msg = [{"role":"user","content":[{"type":"text","text":"".join(parts)}]}]
    out = call_json(session_llm, cfg.models.llm_model, msg, max_tokens=512, temperature=0.0)
    return out if isinstance(out, dict) else None


def _snap_forward_to_heading(text: str, start: int, *, is_tex: bool, limit: int) -> Optional[int]:
    """Snap a boundary forward to the nearest structural chapter heading.

    This is used when anchors were extracted from pages BEFORE unit_start (backward fallback),
    where the "opening" snippet may actually belong to the previous chapter.
    """
    try:
        s = int(max(0, start))
        lim = int(max(0, limit))
        if lim <= 0 or s >= len(text):
            return None
        sub = text[s: min(len(text), s + lim)]
        if not sub:
            return None
        if is_tex:
            pats = [r"\\chapter\*?\{", r"\\section\*?\{", r"\\subsection\*?\{", r"\\subsubsection\*?\{", r"\\part\*?\{", r"\\begin\{chapter\}"]
        else:
            pats = [r"(?im)^\s*chapter\s+\d+\b", r"(?im)^\s*CHAPTER\s+\d+\b"]
        for pat in pats:
            m = re.search(pat, sub)
            if m:
                return s + int(m.start())
        return None
    except Exception:
        return None

def boundaries_from_tex_headings(text: str, pdf_chapters: List[Dict[str, Any]], cfg, logger: logging.Logger) -> Tuple[List[Boundary], List[str]]:
    """
    Use TeX \\chapter{...} boundaries, and align TeX chapters to PDF chapters by title similarity.
    """
    fallbacks: List[str] = []
    # Prefer explicit TeX chapter boundaries; if absent, fall back to chapter-like sections.
    toc_titles = [str(ch.get("title_corrected") or ch.get("title") or "") for ch in pdf_chapters if isinstance(ch, dict)]
    events = parse_tex_chapters(text, toc_titles=toc_titles, toc_title_promotion_min_sim=float(getattr(cfg.align, "tex_toc_title_promotion_min_sim", 0.72)))
    # Keep kind/no for richer downstream tracing.
    tex_events = [(e.start, e.title, e.kind, (e.no if isinstance(e.no, int) else None)) for e in events]
    if not tex_events:
        return [], ["no_tex_headings"]

    # NOTE: Chapter.no is an int. When unknown, fall back to a monotone index-based number
    # so the sequence aligner can still leverage number-consistency heuristics.
    tex_chs = [Ch(no=int((no if (isinstance(no, int) and no > 0) else (i + 1))), title=t)
               for i, (_s, t, _k, no) in enumerate(tex_events)]
    pdf_chs = [Ch(no=int(ch.get("no", i+1) or i+1), title=str(ch.get("title_corrected") or ch.get("title") or "")) 
               for i, ch in enumerate(pdf_chapters)]

    mapping = align_chapter_sequences(pdf_chs, tex_chs)  # pdf_idx -> tex_idx (sparse)
    # if mapping too sparse, fallback
    if len(mapping) < max(2, int(0.6*len(pdf_chs))):
        fallbacks.append("tex_pdf_mapping_sparse")

    _EN_CHAP_RE = re.compile(r"^(?:CHAPTER|Chapter|Chap\.?|CHAP\.?)\s*\d{1,3}\b")
    _ZH_CHAP_RE = re.compile(r"^第\s*([0-9０-９]+|[零〇○一二两三四五六七八九十百千]{1,8})\s*章\b")

    def _is_numbered_chapter_line(title: str) -> bool:
        t = str(title or "").strip()
        if not t:
            return False
        return bool(_EN_CHAP_RE.match(t) or _ZH_CHAP_RE.match(t))

    numline_conf_floor = float(getattr(cfg.align, "tex_numbered_heading_conf_floor", 0.78) or 0.78)

    bounds: List[Boundary] = []
    used_tex = set()
    for pdf_i, ch in enumerate(pdf_chapters):
        no = int(ch.get("no", pdf_i+1) or (pdf_i+1))
        if pdf_i in mapping:
            tex_i = mapping[pdf_i]
            used_tex.add(tex_i)
            start_raw = tex_events[tex_i][0]
            tex_kind = str(tex_events[tex_i][2] or "chapter")
            # confidence based on title similarity
            sim = title_similarity(str(ch.get("title_corrected") or ch.get("title") or ""), tex_events[tex_i][1])
            try:
                tex_no = tex_events[tex_i][3]
                if int(no) > 0 and isinstance(tex_no, int) and int(tex_no) == int(no) and _is_numbered_chapter_line(tex_events[tex_i][1]):
                    sim = max(float(sim), float(numline_conf_floor))
            except Exception:
                pass
            if sim < float(cfg.align.title_match_min_ratio):
                fallbacks.append(f"low_title_sim_ch{no}:{sim:.2f}")
            bounds.append(Boundary(chapter_no=no, start_raw=start_raw, conf=sim, method=f"tex_{tex_kind}"))
        else:
            # allow pipeline-level fallback later
            fallbacks.append(f"pdf_chapter_unmapped_{no}")
            bounds.append(Boundary(chapter_no=no, start_raw=-1, conf=0.0, method="tex_unmapped"))
    # repair unmapped by nearest unused tex chapters if counts match
    return bounds, fallbacks

def boundaries_by_anchor_search(
    session_llm,
    text_path: Path,
    raw_text: str,
    chapters: List[Dict[str, Any]],
    cfg,
    cache_dir: Path,
    logger: logging.Logger,
    *,
    unit_count: Optional[int] = None,
) -> Tuple[List[Boundary], List[str]]:
    """
    General alignment by searching anchors/snippets in normalized shadow text.
    """
    fallbacks: List[str] = []
    cache_path = cache_dir / "text_boundaries.json"
    cached = load_json(cache_path)
    if cached and isinstance(cached, dict) and cached.get("boundaries"):
        try:
            if int(cached.get("_cache_version", 0) or 0) == TEXT_ALIGN_CACHE_VERSION:
                bounds = [Boundary(**b) for b in cached["boundaries"]]
                return bounds, cached.get("fallbacks", [])
        except Exception:
            pass

    # Use a TeX "shadow" text for matching so formulas/macros do not dominate similarity.
    match_text = raw_text
    if _is_tex(text_path):
        try:
            match_text = build_tex_shadow_same_len(raw_text)
        except Exception:
            match_text = raw_text

    norm, norm2raw = normalize_text_for_match(
        match_text,
        lowercase=bool(cfg.text.normalize.lowercase),
        collapse_whitespace=bool(cfg.text.normalize.collapse_whitespace),
        remove_hyphen_linebreak=bool(cfg.text.normalize.remove_hyphen_linebreak),
        normalize_quotes=bool(cfg.text.normalize.normalize_quotes),
        normalize_nfkc=bool(getattr(cfg.text.normalize, "normalize_nfkc", True)),
        remove_soft_hyphen=bool(getattr(cfg.text.normalize, "remove_soft_hyphen", True)),
    )
    chunks = split_into_chunks(norm, int(cfg.text.chunk_size_chars), int(cfg.text.chunk_overlap_chars))

    # Monotone forward search: after we commit a chapter boundary, constrain
    # subsequent searches to later parts of the normalized text.
    last_norm_pos = 0
    last_raw_pos = -1
    bounds: List[Boundary] = []

    # Local band search around an estimated position (from PDF unit_start) dramatically reduces
    # false hits on TOC/Index and reduces LLM fallback rate.
    band_init = int(getattr(cfg.align, "band_initial_chars", 80000) or 80000)
    band_max = int(getattr(cfg.align, "band_max_chars", 300000) or 300000)
    band_expand = float(getattr(cfg.align, "band_expand_factor", 1.8) or 1.8)

    for idx, ch in enumerate(chapters):
        no = int(ch.get("no", idx+1) or (idx+1))
        title = str(ch.get("title_corrected") or ch.get("title") or "")
        anchors = ch.get("anchors") or []
        opening_snippet = str(ch.get("opening_snippet") or "")

        # If anchor extraction had to fall back to pages BEFORE unit_start, the extracted
        # opening_snippet/anchors can actually come from the tail of the previous chapter.
        # In that case, choosing a boundary *at or before* the matched snippet is risky:
        # it may pull previous-chapter content into the current chapter file.
        # We therefore (a) avoid defaulting to candidate-window *starts* and (b) optionally
        # snap forward to the nearest structural chapter heading when the anchor came from
        # a backward page.
        is_backward_anchor = False
        try:
            au = ch.get("anchor_unit")
            us = ch.get("unit_start")
            if au is not None and us is not None:
                is_backward_anchor = int(au) < int(us)
        except Exception:
            is_backward_anchor = False

        # build query
        query_parts = [f"chapter {no}", title] + anchors[:2]
        query = " ".join([p for p in query_parts if p]).strip()
        query_norm = " ".join(tokenize_simple(query))[:500]
        # Choose a search region. Prefer a band around estimated raw position.
        raw_len = len(match_text)
        est_raw: Optional[int] = None
        if unit_count and ch.get("unit_start") is not None:
            try:
                u0 = int(ch.get("unit_start"))
                est_raw = int((u0 / max(1, int(unit_count))) * raw_len)
            except Exception:
                est_raw = None

        radius = band_init
        band_used = False
        scored: List[Tuple[float, int, int]] = []
        while True:
            lo_raw = 0
            hi_raw = raw_len
            if est_raw is not None:
                band_used = True
                lo_raw = max(0, est_raw - radius)
                hi_raw = min(raw_len, est_raw + radius)
                if last_raw_pos >= 0:
                    lo_raw = max(lo_raw, last_raw_pos + 1)

                # Map raw band -> norm band using the monotone norm2raw array.
                lo_norm = bisect.bisect_left(norm2raw, lo_raw)
                hi_norm = bisect.bisect_left(norm2raw, hi_raw)
            else:
                lo_norm = last_norm_pos
                hi_norm = len(norm)

            scored.clear()
            for (s, e, t) in chunks:
                if e <= last_norm_pos:
                    continue
                if e < lo_norm or s > hi_norm:
                    continue
                sim = partial_ratio(query_norm, t)
                scored.append((sim, s, e))

            if scored or (est_raw is None):
                break
            if radius >= band_max:
                break
            radius = min(band_max, int(radius * band_expand))
        scored.sort(reverse=True, key=lambda x: x[0])
        if not scored:
            # Degenerate case: last_norm_pos already beyond end.
            fallbacks.append(f"no_chunk_candidate_ch{no}")
            start_raw = max(0, bounds[-1].start_raw + 1) if bounds else 0
            bounds.append(Boundary(chapter_no=no, start_raw=start_raw, conf=0.0, method="no_candidate"))
            continue

        top_chunks = scored[:max(5, int(cfg.align.topk_candidates)//2)]
        # within each top chunk, try exact occurrences of title tokens / anchors
        cand_norm_positions: List[int] = []
        title_norm = " ".join(tokenize_simple(title))
        if len(title_norm) >= 6:
            for _sim, s, e in top_chunks:
                seg = norm[s:e]
                occ = find_all(seg, title_norm, limit=10)
                for o in occ:
                    cand_norm_positions.append(s + o)
        # also use first anchor token subset
        if anchors:
            a0 = " ".join(tokenize_simple(str(anchors[0])))
            if len(a0) >= 10:
                for _sim, s, e in top_chunks:
                    seg = norm[s:e]
                    occ = find_all(seg, a0[:80], limit=10)
                    for o in occ:
                        cand_norm_positions.append(s + o)

        # NEW: opening_snippet token-seed fallback (robust when TOC title differs from body heading)
        if (not cand_norm_positions) and opening_snippet:
            seed_tokens = [t for t in tokenize_simple(opening_snippet) if len(t) >= 5]
            # keep unique, cap
            uniq: List[str] = []
            seen: set = set()
            for t in seed_tokens:
                if t in seen:
                    continue
                seen.add(t)
                uniq.append(t)
                if len(uniq) >= 10:
                    break
            seed_tokens = uniq

            hits: List[int] = []
            if seed_tokens:
                for _sim, s, e in top_chunks:
                    seg = norm[s:e]
                    for tok in seed_tokens[:8]:
                        occ = find_all(seg, tok, limit=12)
                        for o in occ:
                            hits.append(s + o)

            if hits:
                bin_w = int(getattr(cfg.align, "seed_bin_width_chars", 200) or 200)
                bins: Dict[int, int] = {}
                for p in hits:
                    b = (int(p) // bin_w) * bin_w
                    bins[b] = bins.get(b, 0) + 1
                top_bins = sorted(bins.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
                for b, _cnt in top_bins:
                    cand_norm_positions.append(int(b))
                fallbacks.append(f"seed_tokens_ch{no}")

        # de-dup and sort candidates
        if cand_norm_positions:
            cand_norm_positions = sorted(list({int(p) for p in cand_norm_positions if isinstance(p, int)}))

        # NEW: when we fail to find exact token occurrences (often due to CJK normalization, hyphenation,
        # or TOC/body title mismatch), locate an approximate match position within the top chunks using
        # partial_ratio_alignment and use its dest_start as a better pos_hint.
        if not cand_norm_positions:
            try:
                from rapidfuzz.fuzz import partial_ratio_alignment
            except Exception:
                partial_ratio_alignment = None
            try:
                q_src = ""
                if opening_snippet and len(opening_snippet.strip()) >= 24:
                    q_src = opening_snippet
                elif anchors:
                    q_src = str(anchors[0] or "")
                q_src = (q_src or "").strip()
                if partial_ratio_alignment is not None and q_src:
                    qn, _ = normalize_text_for_match(
                        q_src,
                        lowercase=bool(cfg.text.normalize.lowercase),
                        collapse_whitespace=bool(cfg.text.normalize.collapse_whitespace),
                        remove_hyphen_linebreak=bool(cfg.text.normalize.remove_hyphen_linebreak),
                        normalize_quotes=bool(cfg.text.normalize.normalize_quotes),
                        normalize_nfkc=bool(getattr(cfg.text.normalize, "normalize_nfkc", True)),
                        remove_soft_hyphen=bool(getattr(cfg.text.normalize, "remove_soft_hyphen", True)),
                    )
                    qn = (qn or "").strip()
                    if len(qn) > 1800:
                        qn = qn[:1800]
                    min_score = float(getattr(cfg.align, "opening_align_min_score", 55.0) or 55.0)
                    if min_score <= 1.0:
                        min_score = min_score * 100.0

                    hits: List[Tuple[float, int]] = []
                    for _sim, s, e in top_chunks[:10]:
                        seg = norm[s:e]
                        if not seg:
                            continue
                        al = partial_ratio_alignment(qn, seg)
                        sc = float(getattr(al, "score", 0.0) or 0.0) if al else 0.0
                        ds = int(getattr(al, "dest_start", -1) or -1) if al else -1
                        if al and sc >= min_score and ds >= 0:
                            hits.append((sc, int(s + ds)))

                    if hits:
                        # de-dup by bins to avoid many near-identical hits
                        bin_w = int(getattr(cfg.align, "seed_bin_width_chars", 200) or 200)
                        bins: Dict[int, Tuple[float, int]] = {}
                        for sc, p in hits:
                            b = (int(p) // bin_w) * bin_w
                            cur = bins.get(b)
                            if cur is None or float(sc) > float(cur[0]):
                                bins[b] = (float(sc), int(p))
                        best_bins = sorted(bins.items(), key=lambda kv: (-float(kv[1][0]), kv[0]))[:8]
                        for _b, (_sc, p) in best_bins:
                            cand_norm_positions.append(int(p))
                        cand_norm_positions = sorted(list({int(p) for p in cand_norm_positions}))
                        fallbacks.append(f"opening_align_poshint_ch{no}")
            except Exception:
                pass

        # fallback: use chunk starts
        if not cand_norm_positions:
            cand_norm_positions = [s for _, s, _ in top_chunks[:10]]
            fallbacks.append(f"no_exact_occurrence_ch{no}")

        # score candidates
        candidates: List[Tuple[int, int, str]] = []
        # keep a "pos_hint" for each candidate: the raw position mapped from the norm match.
        # This is typically much closer to the true match location than the candidate window start.
        pos_hints: List[int] = []
        for pos in cand_norm_positions[:int(cfg.align.topk_candidates)]:
            # build excerpt around candidate
            raw_pos = norm2raw[min(pos, len(norm2raw)-1)]
            lo = max(0, raw_pos - int(cfg.align.candidate_window_chars)//2)
            hi = min(len(raw_text), lo + int(cfg.align.candidate_window_chars))
            excerpt = match_text[lo:hi]
            candidates.append((lo, hi, excerpt))
            pos_hints.append(int(raw_pos))

        # LLM refinement with opening snippet (more robust than title alone)
        # LLM refinement uses the same-length shadow text; offsets remain valid for raw_text.
        if not candidates:
            fallbacks.append(f"no_candidate_offsets_ch{no}")
            start_raw = max(0, bounds[-1].start_raw + 1) if bounds else 0
            bounds.append(Boundary(chapter_no=no, start_raw=start_raw, conf=0.0, method="no_candidate_offsets"))
            continue

        refined = _llm_refine(session_llm, cfg, match_text, opening_snippet, candidates)
        if refined and isinstance(refined, dict):
            # Accept an LLM offset only if it falls inside any candidate excerpt window.
            # If the model outputs an offset relative to the chosen candidate, convert it.
            uc = refined.get("used_candidate")
            bo = refined.get("best_offset")
            cand_ranges = [(int(lo), int(hi)) for (lo, hi, _ex) in candidates]
            best_off = None
            try:
                if bo is not None:
                    bo_i = int(bo)
                    if isinstance(uc, int) and 0 <= uc < len(cand_ranges):
                        lo_i, hi_i = cand_ranges[uc]
                        # If bo looks like a relative offset into the candidate, convert it.
                        if 0 <= bo_i < (hi_i - lo_i):
                            bo_i = lo_i + bo_i
                    if any(lo_i <= bo_i < hi_i for lo_i, hi_i in cand_ranges):
                        best_off = int(bo_i)
            except Exception:
                best_off = None

            if best_off is None and isinstance(uc, int) and 0 <= uc < len(cand_ranges):
                # Prefer the raw position hinted by the norm match over window midpoint.
                # Midpoints can land *before* the true match and cause overly-early cuts.
                if uc < len(pos_hints):
                    best_off = int(pos_hints[uc])
                else:
                    lo_i, hi_i = cand_ranges[uc]
                    best_off = int(min(max(lo_i, lo_i + (hi_i - lo_i) // 2), hi_i - 1))
            if best_off is None:
                # Prefer a match-position hint over window start to avoid cutting too early.
                best_off = int(pos_hints[0] if pos_hints else candidates[0][0])

            conf = float(refined.get("confidence", 0.0) or 0.0)
            conf = max(0.0, min(1.0, conf))
            method = "llm_refine"
        else:
            # fallback pick candidate with highest partial match of query
            best_off = int(pos_hints[0] if pos_hints else candidates[0][0])
            best_conf = 0.0
            for (i_c, (off, _hi, ex)) in enumerate(candidates[:10]):
                sim = partial_ratio(query_norm, ex.lower())
                if sim > best_conf:
                    best_conf = sim
                    # Again, prefer the match-position hint rather than the window start.
                    if i_c < len(pos_hints):
                        best_off = int(pos_hints[i_c])
                    else:
                        best_off = int(off)
            conf = float(best_conf)
            method = "local_fallback"
            fallbacks.append(f"llm_refine_failed_ch{no}")

        # Backward-anchor guard (tail-cut rule):
        # If the anchor was extracted from a page before unit_start, the "opening" snippet may
        # actually belong to the tail of the previous chapter. In that case, cutting at the
        # *start* of the matched snippet can pull previous-chapter content into the current
        # chapter file. Instead, cut at the *end* of the matched snippet.
        if is_backward_anchor:
            try:
                from rapidfuzz.fuzz import partial_ratio_alignment
                scan = int(getattr(cfg.align, "backward_anchor_scan_chars", 12000) or 12000)
                scan = max(2000, min(scan, 400000))

                seg_lo = max(0, int(best_off) - scan // 2)
                seg_hi = min(len(match_text), seg_lo + scan)
                segment = match_text[seg_lo:seg_hi]

                # Query normalization is safe here: we only need an approximate tail position.
                q = ""
                if opening_snippet:
                    qn, _ = normalize_text_for_match(opening_snippet)
                    q = (qn or "").lower().strip()
                if len(q) > 1800:
                    q = q[:1800]

                tail = None
                if q and segment:
                    al = partial_ratio_alignment(q, segment.lower())
                    # rapidfuzz partial_ratio_alignment.score is in [0..100].
                    # Allow config to be specified either as ratio (<=1) or score (0..100).
                    min_score = float(getattr(cfg.align, "backward_anchor_tail_min_score", 60.0) or 60.0)
                    if min_score <= 1.0:
                        min_score = min_score * 100.0
                    if al and float(getattr(al, "score", 0.0) or 0.0) >= min_score:
                        tail = seg_lo + int(getattr(al, "dest_end"))

                if tail is not None and int(tail) > int(best_off):
                    best_off = int(tail)
                    method = method + "+backward_anchor_tail"
                    conf = float(min(float(conf or 0.0), 0.88))
                else:
                    # Approximation: if we cannot locate the exact tail, shift forward by a small amount.
                    shift = int(getattr(cfg.align, "backward_anchor_shift_chars", 0) or 0)
                    if shift <= 0 and opening_snippet:
                        shift = min(3000, max(400, len(opening_snippet)))
                    if shift > 0:
                        best_off = min(len(raw_text) - 1, int(best_off) + shift)
                        method = method + "+backward_anchor_tail_approx"
                        conf = float(min(float(conf or 0.0), 0.83))

                # Optional: snap forward to the nearest structural heading to avoid pulling
                # previous-chapter tail into the current chapter when anchors came from
                # pages before unit_start.
                try:
                    snap_lim = int(getattr(cfg.align, "backward_anchor_heading_snap_limit", 40000) or 40000)
                    snap_lim = max(2000, min(snap_lim, 400000))
                    snapped = _snap_forward_to_heading(raw_text, int(best_off), is_tex=_is_tex(text_path), limit=snap_lim)
                    if snapped is not None and int(snapped) > int(best_off):
                        best_off = int(snapped)
                        method = method + "+backward_anchor_heading_snap"
                        conf = float(min(float(conf or 0.0), 0.82))
                except Exception:
                    pass
            except Exception:
                pass

        # Prefer monotonic boundaries; if non-monotonic, pick the nearest candidate strictly after prev. if non-monotonic, pick the nearest candidate strictly after prev.
        if bounds and best_off <= bounds[-1].start_raw:
            prev_raw = int(bounds[-1].start_raw)
            alt = None
            try:
                # Prefer match-position hints for monotone repair.
                pairs = []
                for i_c, (off, _hi, _ex) in enumerate(candidates):
                    hint = int(pos_hints[i_c]) if i_c < len(pos_hints) else int(off)
                    pairs.append((hint, int(off)))
                for hint, off in sorted(pairs, key=lambda x: int(x[0])):
                    if int(hint) > prev_raw:
                        alt = int(hint)
                        break
            except Exception:
                alt = None

            if alt is not None:
                best_off = int(alt)
                try:
                    conf = float(min(conf or 0.0, 0.85))
                except Exception:
                    conf = 0.0
                method = method + "+nonmonotone_next_candidate"
            else:
                fallbacks.append(f"non_monotonic_candidate_ch{no}")
                bounds.append(Boundary(chapter_no=no, start_raw=-1, conf=conf, method="non_monotonic"))
                continue

        # Confidence gate: if too low, leave unresolved to avoid cascading bad splits.
        if conf and conf < float(cfg.align.min_accept_confidence):
            fallbacks.append(f"low_confidence_ch{no}:{conf:.2f}")
            bounds.append(Boundary(chapter_no=no, start_raw=-1, conf=conf, method="low_confidence"))
            continue

        if band_used:
            method = f"{method}_band"

        bounds.append(Boundary(chapter_no=no, start_raw=best_off, conf=conf, method=method))

        # update last_norm_pos (raw->norm via monotone norm2raw)
        last_raw_pos = best_off

        if best_off < len(match_text) and norm2raw:
            try:
                last_norm_pos = max(last_norm_pos, bisect.bisect_left(norm2raw, best_off))
            except Exception:
                pass

    dump_json(cache_path, {"_cache_version": TEXT_ALIGN_CACHE_VERSION, "boundaries":[b.__dict__ for b in bounds], "fallbacks": fallbacks})
    return bounds, fallbacks
