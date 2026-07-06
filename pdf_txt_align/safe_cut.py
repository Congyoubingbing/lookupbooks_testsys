from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import re


_TEX_START_RE = re.compile(
    # Some converted TeX uses "\\subsection *{...}" (space before '*').
    # Allow indentation; capture the TeX command start so we can snap to the true '\\' offset.
    # Prefer explicit [ \t] indentation over \s to avoid surprising matches.
    r"(?m)^[ \t]*(\\(?:chapter|section|subsection|subsubsection|part)\s*\*?\s*\{)"
)


_EN_CHAPTER_RE = re.compile(r"(?i)(?:^|[^a-z])chapter\s*(?:\d+|[ivxlcdm]+)?(?:[^a-z]|$)")
_CN_CHAPTER_RE = re.compile(r"第\s*[0-9０-９零〇○一二两三四五六七八九十百千]+\s*章")

_TEX_DECIMAL_HEADING_RE = re.compile(
    r"(?i)^\\(?:chapter|section|subsection|subsubsection)\s*\*?\s*\{\s*[0-9]{1,3}\s*[.．。]\s*[0-9]{1,3}\b"
)
_TEX_INTEGER_CHAPTERLIKE_RE = re.compile(
    r"(?i)^\\(?:chapter|section|part)\s*\*?\s*\{\s*[0-9]{1,3}(?!\s*[.．。]\s*[0-9])\b"
)
_PLAIN_INTEGER_CHAPTERLIKE_RE = re.compile(
    r"^\s*[0-9]{1,3}(?!\s*[.．。]\s*[0-9])\b(?:\s+|\s*[:：-]\s*)"
)

def _is_strong_chapter_heading_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if _EN_CHAPTER_RE.search(s):
        return True
    if _CN_CHAPTER_RE.search(s):
        return True
    if _TEX_INTEGER_CHAPTERLIKE_RE.match(s):
        return True
    if _TEX_DECIMAL_HEADING_RE.match(s):
        return False
    if _PLAIN_INTEGER_CHAPTERLIKE_RE.match(s):
        return True
    return False


def _is_line_start_or_indented(text: str, pos: int) -> bool:
    """Whether `pos` is at a line start, allowing indentation."""
    if pos <= 0:
        return True
    ln = text.rfind("\n", 0, pos)
    if ln < 0:
        return True
    between = text[ln + 1:pos]
    return (between.strip() == "")




def _cuts_tex_command(full_text: str, pos: int) -> bool:
    if pos <= 0 or pos >= len(full_text):
        return False
    lo = max(0, pos - 24)
    hi = min(len(full_text), pos + 24)
    sub = full_text[lo:hi]
    rel = pos - lo
    for m in re.finditer(r"\\[A-Za-z]+\*?", sub):
        if int(m.start()) < rel < int(m.end()):
            return True
    if rel > 0 and sub[rel - 1] == "\\" and rel < len(sub) and sub[rel].isalpha():
        return True
    return False

def _is_in_math(tex: str, pos: int) -> bool:
    r"""Best-effort detection for being inside common TeX math regions.

    This is heuristic (no full TeX parsing), but it prevents the most damaging
    cut points (inside $...$ or \[...\] or equation environments).
    """
    if pos <= 0:
        return False
    lo = max(0, pos - 6000)
    snippet = tex[lo:pos]

    # Unescaped $ count parity
    dollars = re.findall(r"(?<!\\)\$", snippet)
    if len(dollars) % 2 == 1:
        return True

    # \[ ... \] parity
    if snippet.count("\\[") > snippet.count("\\]"):
        return True

    # Common equation-like environments
    envs = [
        "equation",
        "equation*",
        "align",
        "align*",
        "eqnarray",
        "eqnarray*",
        "gather",
        "gather*",
        "multline",
        "multline*",
    ]
    for env in envs:
        if snippet.count(f"\\begin{{{env}}}") > snippet.count(f"\\end{{{env}}}"):
            return True

    return False


def _candidate_positions(tex: str, idx: int, window: int) -> List[int]:
    lo = max(0, idx - window)
    hi = min(len(tex), idx + window)
    sub = tex[lo:hi]

    cands: List[int] = []

    # Prefer explicit structural headings
    for m in _TEX_START_RE.finditer(sub):
        # m.start(1) points to the TeX command start (after indentation)
        pos = lo + m.start(1)
        cands.append(pos)

    # Newline boundary
    nl = tex.rfind("\n", lo, hi)
    if nl >= 0:
        cands.append(nl + 1)

    # Paragraph boundary (double newline)
    dbl = tex.rfind("\n\n", lo, hi)
    if dbl >= 0:
        cands.append(dbl + 2)

    # As a last option, allow original index
    cands.append(idx)

    # De-dup, keep stable order
    seen = set()
    out: List[int] = []
    for p in cands:
        if 0 <= p <= len(tex) and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def snap_boundary(
    text: str,
    idx: int,
    *,
    is_tex: bool,
    window: int = 2000,
    midline_penalty: int = 250,
    strong_chapter_bonus: int = 800,
    generic_heading_bonus: int = 30,
    tex_nonchapter_heading_bonus: int = 10,
    tex_decimal_heading_penalty: int = 140,
    avoid_midword: bool = False,
    midword_penalty: float = 0.35,
    avoid_tex_command_split: bool = True,
    tex_command_penalty: float = 0.8,
) -> Tuple[int, Dict[str, Any]]:
    """Snap a cut boundary to a safer location (line/heading boundary).

    Returns (snapped_idx, info).
    """
    idx = int(idx)
    if idx <= 0 or idx >= len(text):
        return idx, {"snapped": idx, "delta": 0, "rule": "noop"}

    if not is_tex:
        # For plain text, snap to a safe boundary. Prefer newline; if absent (e.g., very long lines),
        # fall back to nearest whitespace to avoid mid-word cuts.
        lo = max(0, idx - window)
        hi = min(len(text), idx + window)
        nl = text.rfind("\n", lo, hi)
        if nl >= 0:
            snapped = nl + 1
            return snapped, {"snapped": snapped, "delta": snapped - idx, "rule": "newline"}

        # Whitespace fallback
        prev_ws = -1
        for j in range(idx - 1, lo - 1, -1):
            if text[j].isspace():
                prev_ws = j
                break
        next_ws = -1
        for j in range(idx, hi):
            if text[j].isspace():
                next_ws = j
                break

        if prev_ws >= 0:
            snapped = prev_ws + 1
            return snapped, {"snapped": snapped, "delta": snapped - idx, "rule": "whitespace_prev"}
        if next_ws >= 0:
            snapped = next_ws + 1
            return snapped, {"snapped": snapped, "delta": snapped - idx, "rule": "whitespace_next"}

        return idx, {"snapped": idx, "delta": 0, "rule": "noop"}

    best = idx
    best_score = 10**18
    best_rule = "idx"
    best_heading = None
    candidates_dbg: List[Dict[str, Any]] = []
    midline_penalty = int(midline_penalty or 0)
    strong_chapter_bonus = int(strong_chapter_bonus or 0)
    generic_heading_bonus = int(generic_heading_bonus or 0)
    tex_nonchapter_heading_bonus = int(tex_nonchapter_heading_bonus or 0)
    midword_penalty = float(midword_penalty or 0.0)
    tex_command_penalty = float(tex_command_penalty or 0.0)

    for p in _candidate_positions(text, idx, window=window):
        # Hard reject: math region
        if _is_in_math(text, p):
            candidates_dbg.append({
                "pos": int(p),
                "rule": "reject_math",
                "score": None,
                "dist": abs(int(p) - int(idx)),
                "penalty": None,
                "heading": None,
                "line_start": _is_line_start_or_indented(text, p),
            })
            continue

        if avoid_tex_command_split and _cuts_tex_command(text, p):
            candidates_dbg.append({
                "pos": int(p),
                "rule": "reject_tex_command_midcut",
                "score": None,
                "dist": abs(int(p) - int(idx)),
                "penalty": None,
                "heading": None,
                "line_start": _is_line_start_or_indented(text, p),
            })
            continue

        dist = abs(p - idx)

        # Penalize mid-line cuts (but treat indentation as line-start)
        penalty = 0
        if not _is_line_start_or_indented(text, p):
            penalty += midline_penalty

        # Penalize mid-word cuts for English text (configurable).
        is_midword = _is_midword_cut(text, p)
        if avoid_midword and is_midword:
            penalty += int(round(1000.0 * max(0.0, midword_penalty)))

        if avoid_tex_command_split and _cuts_tex_command(text, p):
            penalty += int(round(1000.0 * max(0.0, tex_command_penalty)))

        # Prefer explicit headings; prioritize \chapter over \section.
        # Also treat "CHAPTER N" / "第X章" as strong even if the TeX command is \section*.
        heading_cmd = None
        strong_line = False
        is_decimal_tex_heading = False
        if p < len(text) and text[p:p+1] == "\\":
            m = re.match(r"\\(chapter|section|subsection|subsubsection|part)\b", text[p:p+32])
            if m:
                heading_cmd = str(m.group(1))
            is_decimal_tex_heading = bool(_TEX_DECIMAL_HEADING_RE.match(text[p:p+160]))
            nl = text.find("\n", p, min(len(text), p + 400))
            if nl == -1:
                nl = min(len(text), p + 400)
            strong_line = _is_strong_chapter_heading_line(text[p:nl])
            # Historical pitfall: non-chapter TeX headings (e.g., \section{6.1 ...})
            # were given a large bonus and could hijack chapter boundaries. Give a large bonus
            # only to real/strong chapter headings; keep section-level headings as weak hints.
            if heading_cmd in {"chapter", "part"} or strong_line:
                bonus = generic_heading_bonus + strong_chapter_bonus
                rule = "heading_chapter"
            elif heading_cmd in {"section", "subsection", "subsubsection"}:
                bonus = tex_nonchapter_heading_bonus
                rule = "heading_nonchapter"
            else:
                bonus = generic_heading_bonus
                rule = "heading"
            penalty -= int(bonus)
            if is_decimal_tex_heading:
                penalty += int(tex_decimal_heading_penalty or 0)
        elif p > 1 and text[p-2:p] == "\n\n":
            penalty -= 30
            rule = "paragraph"
        elif p > 0 and text[p-1] == "\n":
            penalty -= 10
            rule = "newline"
        else:
            rule = "idx"

        score = dist + penalty
        candidates_dbg.append({
            "pos": int(p),
            "rule": str(rule),
            "score": int(score),
            "dist": int(dist),
            "penalty": int(penalty),
            "heading": heading_cmd,
            "line_start": _is_line_start_or_indented(text, p),
            "strong_chapter_heading": bool(strong_line),
            "midword": bool(is_midword),
            "tex_command_midcut": bool(_cuts_tex_command(text, p)),
            "decimal_tex_heading": bool(is_decimal_tex_heading),
        })
        if score < best_score:
            best_score = score
            best = p
            best_rule = rule
            best_heading = heading_cmd

    # Sort candidates by score (None at end)
    def _cand_key(c: Dict[str, Any]) -> Tuple[int, int]:
        sc = c.get("score")
        if sc is None:
            return (10**18, int(c.get("pos") or 0))
        return (int(sc), int(c.get("pos") or 0))

    if avoid_tex_command_split and _cuts_tex_command(text, best):
        lo2 = max(0, int(best) - 32)
        hi2 = min(len(text), int(best) + 32)
        sub2 = text[lo2:hi2]
        rel2 = int(best) - lo2
        repaired = None
        for m in re.finditer(r"\\[A-Za-z]+\*?", sub2):
            if int(m.start()) < rel2 < int(m.end()):
                repaired = lo2 + int(m.start())
                break
        if repaired is not None:
            best = int(repaired)
            best_rule = "repair_tex_command_left"

    candidates_dbg_sorted = sorted(candidates_dbg, key=_cand_key)
    has_strong = any(bool(c.get("strong_chapter_heading")) for c in candidates_dbg_sorted)
    return best, {
        "snapped": int(best),
        "delta": int(best) - int(idx),
        "rule": str(best_rule),
        "heading": best_heading,
        "has_strong_chapter_heading": bool(has_strong),
        "has_decimal_tex_heading_candidates": any(bool(c.get("decimal_tex_heading")) for c in candidates_dbg_sorted),
        "candidates": candidates_dbg_sorted,
    }


def snap_boundaries(
    text: str,
    boundaries: List[int],
    *,
    is_tex: bool,
    window: int = 2000,
    min_gap: int = 1,
    min_positions: Optional[List[int]] = None,
    midline_penalty: int = 250,
    strong_chapter_bonus: int = 800,
    generic_heading_bonus: int = 30,
    tex_nonchapter_heading_bonus: int = 10,
    tex_decimal_heading_penalty: int = 140,
    avoid_midword: bool = False,
    midword_penalty: float = 0.35,
    avoid_tex_command_split: bool = True,
    tex_command_penalty: float = 0.8,
) -> Tuple[List[int], Dict[str, Any]]:
    """Snap a list of boundaries while maintaining strict monotonicity."""
    report: Dict[str, Any] = {
        "is_tex": bool(is_tex),
        "window": int(window),
        "midline_penalty": int(midline_penalty),
        "strong_chapter_bonus": int(strong_chapter_bonus),
        "generic_heading_bonus": int(generic_heading_bonus),
        "tex_nonchapter_heading_bonus": int(tex_nonchapter_heading_bonus),
        "tex_decimal_heading_penalty": int(tex_decimal_heading_penalty),
        "safe_cut_avoid_midword": bool(avoid_midword),
        "safe_cut_midword_penalty": float(midword_penalty),
        "safe_cut_avoid_tex_command_split": bool(avoid_tex_command_split),
        "safe_cut_tex_command_penalty": float(tex_command_penalty),
        "items": [],
    }
    mg = max(1, int(min_gap or 1))
    report["min_gap"] = mg
    out: List[int] = []
    prev = -1
    for i, b in enumerate(boundaries or []):
        snapped, info = snap_boundary(
            text,
            int(b),
            is_tex=is_tex,
            window=window,
            midline_penalty=midline_penalty,
            strong_chapter_bonus=strong_chapter_bonus,
            generic_heading_bonus=generic_heading_bonus,
            tex_nonchapter_heading_bonus=tex_nonchapter_heading_bonus,
            tex_decimal_heading_penalty=tex_decimal_heading_penalty,
            avoid_midword=avoid_midword,
            midword_penalty=midword_penalty,
            avoid_tex_command_split=avoid_tex_command_split,
            tex_command_penalty=tex_command_penalty,
        )
        if min_positions is not None and i < len(min_positions) and min_positions[i] is not None:
            min_pos = max(0, int(min_positions[i]))
            if int(snapped) < min_pos:
                snapped = min_pos
                info["rule"] = str(info.get("rule")) + "+min_pos"
        
        snapped = max(prev + mg, min(len(text), int(snapped)))
        if snapped != int(info.get("snapped", snapped)):
            info["snapped"] = snapped
            info["delta"] = snapped - int(b)
            info["rule"] = str(info.get("rule")) + "+monotone"
        out.append(snapped)
        prev = snapped
        info.update({"i": i, "orig": int(b)})
        report["items"].append(info)

    # final guard
    for j in range(1, len(out)):
        if out[j] <= out[j-1]:
            out[j] = min(len(text), out[j-1] + mg)
    return out, report


def _is_midword_cut(text: str, pos: int) -> bool:
    if pos <= 0 or pos >= len(text):
        return False
    a = text[pos-1]
    b = text[pos]
    # English-focused: avoid splitting alphabetic words across the cut.
    return a.isalpha() and b.isalpha()

