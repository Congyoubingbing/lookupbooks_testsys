from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .similarity import normalize_title_robust, title_similarity


@dataclass
class HeadingEvent:
    kind: str  # "chapter"|"section"|"subsection"|"part"|"unknown"
    title: str
    no: Optional[int]
    start: int  # char index in raw text
    starred: bool = False


_TEX_CMD_RE = re.compile(r"\\(chapter\*?|section\*?|subsection\*?|part\*?)\{([^}]*)\}")


def _parse_leading_int(title: str) -> Optional[int]:
    # handles "1", "1.", "1 -", "1:"
    m = re.match(r"^\s*(\d{1,3})\b", title)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


_CN_DIGITS = {
    '零': 0, '〇': 0, '○': 0,
    '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9,
}
_CN_UNITS = {'十': 10, '百': 100, '千': 1000}


def _strip_title_noise(title: str) -> str:
    t = str(title or "")
    t = re.sub(r"\\(?:quad|qquad|,|;|!|hspace\*?\s*\{[^{}]*\}|hskip\s*[^\s]+)", " ", t, flags=re.IGNORECASE)
    t = t.replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _looks_all_caps_title(title: str) -> bool:
    t = _strip_title_noise(title)
    letters = [ch for ch in t if ch.isalpha()]
    if len(letters) < 6:
        return False
    return (sum(1 for ch in letters if ch.isupper()) / float(len(letters))) >= 0.8


def _parse_cn_numeral(s: str) -> Optional[int]:
    """Parse very common Chinese numerals used in headings.

    Supports up to 千 and typical combinations: 十/十一/二十/二十三/一百零二...
    Returns None if input is empty or unparseable.
    """
    if not s:
        return None
    s = (s or "").strip()
    if not s:
        return None
    # normalize full-width digits
    s2 = s.translate(str.maketrans({'０': '0', '１': '1', '２': '2', '３': '3', '４': '4', '５': '5', '６': '6', '７': '7', '８': '8', '９': '9'}))
    if s2.isdigit():
        try:
            return int(s2)
        except Exception:
            return None

    total = 0
    num = 0
    unit_seen = False
    for ch in s2:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit_seen = True
            unit = _CN_UNITS[ch]
            if num == 0:
                num = 1
            total += num * unit
            num = 0
        else:
            # unexpected char
            return None
    total += num
    if total == 0 and unit_seen:
        # "十" => 10
        total = 10
    return total if total > 0 else None


def _parse_heading_no(title: str) -> Optional[int]:
    """Parse a heading number from common TeX/plain formats.

    Handles:
      - leading digits: "3", "3." ...
      - English: "CHAPTER 3", "Chap. 3"
      - Chinese: "第3章", "第十章"
    """
    t = _strip_title_noise(title)
    if not t:
        return None
    no = _parse_leading_int(t)
    if no is not None:
        return no
    m = re.match(r"^(?:CHAPTER|Chapter|Chap\.?|CHAP\.?)\s*(\d{1,3})\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m = re.match(r"^第(?:\s|\\quad|\\qquad|\\hspace\*?\s*\{[^{}]*\})*([0-9０-９]{1,3}|[零〇○一二两三四五六七八九十百千]{1,8})(?:\s|\\quad|\\qquad|\\hspace\*?\s*\{[^{}]*\})*章", t)
    if m:
        return _parse_cn_numeral(m.group(1))
    return None


def _find_matching_brace(text: str, open_brace_pos: int, *, max_scan: int = 12000) -> Optional[int]:
    r"""Find the matching '}' for a '{' at open_brace_pos, allowing nested braces.

    - Skips escaped braces like \{ and \}
    - Hard-limits scan length to avoid pathological TeX (or corrupted OCR) blowing up runtime
    """
    if open_brace_pos < 0 or open_brace_pos >= len(text) or text[open_brace_pos] != "{":
        return None
    depth = 0
    i = open_brace_pos
    limit = min(len(text), open_brace_pos + int(max_scan or 0))
    while i < limit:
        ch = text[i]
        if ch == "\\":  # skip escaped character
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


_TEX_CMD_PREFIX_RE = re.compile(
    # allow optional spaces before "*" (some converted TeX uses "\\subsection *{...")
    r"\\(part|chapter|section|subsection|subsubsection)\s*(\*)?\s*(?:\[[^\]]*\]\s*)?\{",
    flags=re.MULTILINE,
)


def parse_tex_headings(text: str, *, dedup_gap_chars: int = 5000) -> List[HeadingEvent]:
    r"""Parse TeX structural headings as boundary candidates.

    More robust than a simple regex:
    - Supports optional short-title argument: \section*[...]{...}
    - Supports nested braces inside {...} (within a scan limit)
    - Supports subsubsection as well
    """

    events: List[HeadingEvent] = []
    for m in _TEX_CMD_PREFIX_RE.finditer(text):
        kind = (m.group(1) or "unknown").strip().lower()
        starred = bool(m.group(2))
        kind = kind if kind in {"chapter", "section", "subsection", "subsubsection", "part"} else "unknown"
        open_brace_pos = m.end() - 1
        close_pos = _find_matching_brace(text, open_brace_pos, max_scan=12000)
        if close_pos is None:
            continue
        title = (text[m.end():close_pos] or "").strip()
        if not title:
            continue
        no = _parse_heading_no(title)
        events.append(HeadingEvent(kind=kind, title=title, no=no, start=m.start(), starred=starred))

    if not events:
        return events

    # Stable order + light dedup: TeX sources (especially OCR/converted) can emit
    # duplicated structural commands near TOC/front-matter.
    events.sort(key=lambda e: e.start)
    dedup_gap_chars = max(0, int(dedup_gap_chars or 0))

    def _norm_title(t: str) -> str:
        tl = (t or "").lower().strip()
        tl = re.sub(r"\s+", " ", tl)
        tl = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", tl)
        return tl

    out: List[HeadingEvent] = []
    last = None
    last_norm = ""
    for e in events:
        n = _norm_title(e.title)
        if last is not None and e.kind == last.kind and n and n == last_norm and (e.start - last.start) <= dedup_gap_chars:
            continue
        out.append(e)
        last = e
        last_norm = n

    return out




def is_frontmatter_title(title: str) -> bool:
    """Heuristic filter for headings that belong to front/back matter (目录/索引/参考文献等).

    Used to make TeX-derived fallback plans more conservative.
    """
    if not title:
        return False
    t = title.strip()
    tl = t.lower()
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
        if kw in tl:
            return True
    # Chinese common
    cjk_kws = ["目录", "前言", "序言", "序", "致谢", "参考文献", "索引", "符号表", "图目录", "表目录"]
    for kw in cjk_kws:
        if kw in t:
            return True
    return False

def parse_tex_chapters(text: str, *, min_chapters: int = 3, dedup_gap_chars: int = 5000, toc_titles: Optional[List[str]] = None, toc_title_promotion_min_sim: float = 0.72) -> List[HeadingEvent]:
    r"""Backwards-compatible: return chapter-like events.

    Primary: \chapter / \chapter*
    Fallback: treat chapter-like \section / \section* as chapter boundaries
              when \chapter is absent or clearly incomplete.
    """

    events = parse_tex_headings(text, dedup_gap_chars=dedup_gap_chars)
    ch = [e for e in events if e.kind == "chapter" and not is_frontmatter_title(e.title)]
    sec = [e for e in events if e.kind == "section" and not is_frontmatter_title(e.title)]
    sub = [e for e in events if e.kind == "subsection" and not is_frontmatter_title(e.title)]

    # Merge OCR pattern: \chapter{5} followed by \section*{Other Polymer Systems}
    _merged_ch: List[HeadingEvent] = []
    _all_sorted = sorted(events, key=lambda e: e.start)
    _i = 0
    while _i < len(_all_sorted):
        _e = _all_sorted[_i]
        if _e.kind == "chapter" and (not is_frontmatter_title(_e.title)):
            _no = _parse_heading_no(_e.title)
            _chapter_is_number_only = bool(re.fullmatch(r"[0-9]{1,3}", _strip_title_noise(_e.title) or ""))
            if _chapter_is_number_only and (_i + 1) < len(_all_sorted):
                _nxt = _all_sorted[_i + 1]
                if _nxt.kind == "section" and bool(getattr(_nxt, "starred", False)) and (not is_frontmatter_title(_nxt.title)):
                    _nxt_t = _strip_title_noise(_nxt.title)
                    if _parse_heading_no(_nxt_t) is None and not re.match(r"^\d{1,3}\s*\.", _nxt_t):
                        _merged_ch.append(HeadingEvent(kind="chapter", title=_nxt.title, no=_no, start=_e.start, starred=True))
                        _i += 2
                        continue
            _merged_ch.append(_e)
        _i += 1
    if _merged_ch:
        ch = _merged_ch

    toc_norms = [normalize_title_robust(t) for t in (toc_titles or []) if str(t or '').strip()]

    def _toc_promoted(title: str) -> bool:
        if not toc_norms:
            return False
        tn = normalize_title_robust(title or "")
        if not tn:
            return False
        try:
            best = max(float(title_similarity(tn, x)) for x in toc_norms)
        except Exception:
            best = 0.0
        return best >= float(toc_title_promotion_min_sim)

    def _looks_like_chapter_title(title: str, *, starred: bool = False) -> bool:
        t = _strip_title_noise(title)
        if not t:
            return False
        if re.match(r"^\d+\.\d+(?:\.\d+)*\b", t):
            return False
        if re.match(r"^\d{1,3}\s*\.\s+", t) and not re.match(r"^(?:CHAPTER|Chapter|Chap\.?|CHAP\.?)\s*\d+\b", t):
            return False
        if re.match(r"^(?:REFERENCES?|BIBLIOGRAPHY|SUBJECT\s+INDEX|AUTHOR\s+INDEX|INDEX|APPENDIX\b)", t, flags=re.I):
            return False
        if re.match(r"^第(?:\s|\\quad|\\qquad|\\hspace\*?\s*\{[^{}]*\})*([0-9０-９]{1,3}|[零〇○一二两三四五六七八九十百千]{1,8})(?:\s|\\quad|\\qquad|\\hspace\*?\s*\{[^{}]*\})*章", t):
            return True
        if re.match(r"^(?:CHAPTER|Chapter|Chap\.?|CHAP\.?)\s*\d+\b", t):
            return True
        if starred and _looks_all_caps_title(t):
            return True
        if _parse_leading_int(t) is not None and len(t) <= 64 and not re.match(r"^\d+\.\d", t):
            return True
        if _toc_promoted(t):
            return True
        return False

    chapterlike_sec = [e for e in sec if _looks_like_chapter_title(e.title, starred=bool(getattr(e, "starred", False)))]
    chapterlike_sub = [e for e in sub if _looks_like_chapter_title(e.title, starred=bool(getattr(e, "starred", False)))]

    # If \chapter exists and looks complete, we usually keep it — but some OCR TeX files
    # switch to chapter-like \section* headings for later chapters (e.g., "CHAPTER 7: ...").
    # Detect such missing chapter-like headings and merge them instead of returning too early.
    min_ch = max(1, int(min_chapters or 1))
    if ch and len(ch) >= min_ch:
        gap0 = max(0, int(dedup_gap_chars or 0))
        ch_positions = [int(e.start) for e in ch]
        ch_nos = {int(e.no) for e in ch if isinstance(getattr(e, "no", None), int)}

        def _near_existing_ch(pos: int) -> bool:
            for p in ch_positions:
                if abs(int(pos) - int(p)) <= gap0:
                    return True
            return False

        missing_chapterlike = False
        for e in sorted((chapterlike_sec + chapterlike_sub), key=lambda x: x.start):
            if _near_existing_ch(int(e.start)):
                continue
            eno = _parse_heading_no(e.title)
            if isinstance(eno, int) and eno > 0 and eno not in ch_nos:
                missing_chapterlike = True
                break
            if _toc_promoted(e.title):
                missing_chapterlike = True
                break

        if not missing_chapterlike:
            return ch

    # If \chapter exists but is sparse/incomplete, KEEP it and merge in chapter-like sections.
    # This avoids dropping true \chapter boundaries when front-matter or mixed formatting is present.
    if ch:
        # Start with true \chapter events, then add chapter-like \section/\subsection
        # only when they are not too close to an existing boundary.
        gap = max(0, int(dedup_gap_chars or 0))
        out: List[HeadingEvent] = sorted(list(ch), key=lambda e: e.start)

        def _too_close(pos: int) -> bool:
            for e in out:
                if abs(int(pos) - int(e.start)) <= gap:
                    return True
            return False

        for e in sorted((chapterlike_sec + chapterlike_sub), key=lambda x: x.start):
            if _too_close(e.start):
                continue
            out.append(e)
        out.sort(key=lambda e: e.start)

        # If chapter-like sections dominate and chapters are extremely few, allow using them directly.
        if (not out) and (chapterlike_sec or chapterlike_sub):
            out = (chapterlike_sec or chapterlike_sub)
        return out

    # No \chapter: prefer chapter-like sections/subsections when available.
    if chapterlike_sec and len(chapterlike_sec) >= min_ch:
        return chapterlike_sec
    if chapterlike_sub and len(chapterlike_sub) >= min_ch:
        return chapterlike_sub
    # Plain-text fallback for OCR/converted "TeX" where headings are present as text
    # (e.g., "第4章 ..." / "CHAPTER 7 ..."), but not as LaTeX commands.
    if not ch and not chapterlike_sec and not chapterlike_sub:
        plain = [e for e in parse_plain_chapters(text) if not is_frontmatter_title(e.title)]

        # light dedup: same title within a short char distance
        out: List[HeadingEvent] = []
        last_pos = -10**9
        last_norm = ""
        # HeadingEvent uses the field name `start` (char index). Keep this consistent
        # to avoid silently disabling plain-text chapter fallback.
        for e in sorted(plain, key=lambda x: x.start):
            n = re.sub(r"\s+", " ", (e.title or "").strip().lower())
            if n and n == last_norm and (e.start - last_pos) < max(1000, int(dedup_gap_chars * 0.6)):
                continue
            out.append(e)
            last_pos = e.start
            last_norm = n

        if out and len(out) >= max(1, int(min_chapters or 1)):
            return out

    # Otherwise: if we cannot reliably infer chapter-level boundaries, return []
    # (Let the caller decide stronger fallbacks; do NOT treat all sections as chapters.)
    if ch:
        return ch
    return []

def parse_plain_chapters(text: str) -> List[HeadingEvent]:
    """Parse plain text chapter headings like:

      CHAPTER 1 Title
      Chapter 1 Title
      第1章 标题
    """

    events: List[HeadingEvent] = []
    patterns = [
        r"^(CHAPTER)\s+(\d+)\s*[:.\-]?\s*(.+)$",
        r"^(Chapter)\s+(\d+)\s*[:.\-]?\s*(.+)$",
        r"^第\s*([0-9０-９]+|[零〇○一二两三四五六七八九十百千]{1,8})\s*章\s*(.+)$",
    ]
    lines = text.splitlines(True)
    idx = 0
    for ln in lines:
        stripped = ln.strip()
        for pat in patterns:
            m = re.match(pat, stripped)
            if m:
                if "CHAPTER" in m.group(1) or "Chapter" in m.group(1):
                    no = int(m.group(2))
                    title = m.group(3).strip()
                else:
                    raw = m.group(1)
                    raw = raw.translate(str.maketrans({'０':'0','１':'1','２':'2','３':'3','４':'4','５':'5','６':'6','７':'7','８':'8','９':'9'}))
                    if raw.isdigit():
                        no = int(raw)
                    else:
                        # very small Chinese numeral support: 十/十一/二十/... up to 千
                        cn_map = {'零':0,'〇':0,'○':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
                        unit_map = {'十':10,'百':100,'千':1000}
                        total = 0; num = 0; unit_seen=False
                        for ch in raw:
                            if ch in cn_map: num = cn_map[ch]
                            elif ch in unit_map:
                                unit_seen=True; unit=unit_map[ch];
                                if num==0: num=1
                                total += num*unit; num=0
                        total += num
                        if total==0 and unit_seen: total=10
                        no = total if total>0 else 0
                    title = m.group(2).strip()
                events.append(HeadingEvent(kind="chapter", title=title, no=no, start=idx))
                break
        idx += len(ln)
    return events


def build_tex_shadow_same_len(tex: str) -> str:
    """Return a "shadow" TeX string with math/commands stripped but length preserved.

    This is critical for robust matching:
    - TeX contains dense math, commands, references that do not appear in scanned PDF OCR.
    - We match on the shadow text (mostly narrative), but output raw TeX unchanged.

    The returned string has EXACTLY the same length as the input, so all character offsets
    remain valid for the raw TeX.
    """

    s = list(tex)
    n = len(s)
    i = 0

    def _blank(lo: int, hi: int):
        for k in range(max(0, lo), min(n, hi)):
            # preserve newlines to keep rough structure
            if s[k] not in "\r\n":
                s[k] = " "

    # Remove comments: % ... endline
    for m in re.finditer(r"%[^\n]*", tex):
        _blank(m.start(), m.end())

    # Remove math environments and inline math
    math_spans = []
    math_spans += [(m.start(), m.end()) for m in re.finditer(r"\$\$.*?\$\$", tex, flags=re.S)]
    math_spans += [(m.start(), m.end()) for m in re.finditer(r"\$.*?\$", tex, flags=re.S)]
    math_spans += [(m.start(), m.end()) for m in re.finditer(r"\\\[.*?\\\]", tex, flags=re.S)]
    math_spans += [(m.start(), m.end()) for m in re.finditer(r"\\\(.*?\\\)", tex, flags=re.S)]
    for lo, hi in math_spans:
        _blank(lo, hi)

    # Remove common math environments: \begin{equation}...\end{equation} etc.
    env_names = [
        "equation", "align", "align*", "eqnarray", "gather", "multline",
        "split", "cases", "math", "displaymath",
    ]
    for env in env_names:
        pat = rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}"
        for m in re.finditer(pat, tex, flags=re.S):
            _blank(m.start(), m.end())

    # Replace TeX commands (backslash words) with spaces, keep braces content
    for m in re.finditer(r"\\[a-zA-Z]+\*?", tex):
        _blank(m.start(), m.end())

    # Remove \label{...} \ref{...} \cite{...} blocks but keep braces length
    for m in re.finditer(r"\\(label|ref|eqref|cite|citep|citet)\{[^}]*\}", tex):
        _blank(m.start(), m.end())

    return "".join(s)


def normalize_text_for_match(
    text: str,
    *,
    lowercase: bool = True,
    collapse_whitespace: bool = True,
    remove_hyphen_linebreak: bool = True,
    normalize_quotes: bool = True,
    normalize_nfkc: bool = True,
    remove_soft_hyphen: bool = True,
) -> Tuple[str, List[int]]:
    """Build a normalized "shadow" text and a mapping from normalized index -> raw index.

    This function is used for *matching only* and does NOT change raw outputs.

    Notes:
    - Some PDFs/TXT/TeX contain Unicode ligatures (ﬁ/ﬂ/ﬃ/ﬄ) or compatibility glyphs.
      We expand them into ASCII sequences while mapping all expanded chars back to the same raw index.
    - Optional NFKC normalization is applied per-character (can expand into multiple chars).
    """

    norm_chars: List[str] = []
    norm_to_raw: List[int] = []

    # Common ligatures (Unicode Presentation Forms)
    lig_map = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }

    i = 0
    while i < len(text):
        ch = text[i]

        # Remove soft hyphen (often inserted by PDF text layer / OCR)
        if remove_soft_hyphen and ch == "\u00ad":
            i += 1
            continue

        # remove hyphen + linebreak (common in OCR): "poly-\nmer" -> "polymer"
        if remove_hyphen_linebreak and ch in ("-", "‐", "‑", "‒", "–") and i + 1 < len(text) and text[i + 1] in "\r\n":
            i += 1
            while i < len(text) and text[i] in "\r\n":
                i += 1
            continue

        if normalize_quotes:
            if ch in "“”„‟":
                ch = '"'
            elif ch in "‘’‚‛":
                ch = "'"

        # Expand ligatures
        if ch in lig_map:
            exp = lig_map[ch]
            # Apply lowercase after expansion
            if lowercase:
                exp = exp.lower()
            for c2 in exp:
                norm_chars.append(c2)
                norm_to_raw.append(i)
            i += 1
            continue

        # Apply NFKC per character (may expand)
        if normalize_nfkc:
            try:
                ch_nfkc = unicodedata.normalize("NFKC", ch)
            except Exception:
                ch_nfkc = ch
        else:
            ch_nfkc = ch

        # If normalization expanded, map each resulting char back to the same raw index.
        if ch_nfkc and len(ch_nfkc) > 1:
            exp = ch_nfkc
            if lowercase:
                exp = exp.lower()
            for c2 in exp:
                if collapse_whitespace and c2.isspace():
                    # normalize whitespace expansions into a single space
                    raw_i = i
                    if norm_chars and norm_chars[-1] != " ":
                        norm_chars.append(" ")
                        norm_to_raw.append(raw_i)
                    continue
                norm_chars.append(c2)
                norm_to_raw.append(i)
            i += 1
            continue

        ch2 = ch_nfkc if ch_nfkc else ch

        if lowercase:
            ch2 = ch2.lower()

        if collapse_whitespace and ch2.isspace():
            raw_i = i
            while i < len(text) and text[i].isspace():
                i += 1
            if norm_chars and norm_chars[-1] != " ":
                norm_chars.append(" ")
                norm_to_raw.append(raw_i)
            continue

        norm_chars.append(ch2)
        norm_to_raw.append(i)
        i += 1

    return "".join(norm_chars), norm_to_raw


def split_into_chunks(norm_text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    n = len(norm_text)
    step = max(1, chunk_size - overlap)
    for start in range(0, n, step):
        end = min(n, start + chunk_size)
        out.append((start, end, norm_text[start:end]))
        if end >= n:
            break
    return out
