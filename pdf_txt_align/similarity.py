from __future__ import annotations
from typing import List, Set

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

import difflib
import re


_LATEX_CMD_WITH_ARG_RE = re.compile(
    r"\\[a-zA-Z@]+\s*\*?\s*(?:\[[^\]]*\]\s*)?\{([^\{\}]*)\}",
    flags=re.UNICODE,
)
_LATEX_CMD_RE = re.compile(
    r"\\[a-zA-Z@]+\s*\*?(?:\s*\[[^\]]*\])?(?:\s*\{[^\}]*\})?",
    flags=re.UNICODE,
)
_PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", flags=re.UNICODE)


def normalize_title_robust(s: str) -> str:
    """Normalize titles for matching.

    - removes common LaTeX commands
    - lowercases
    - strips punctuation
    - collapses whitespace
    """
    s = (s or "").strip()
    if not s:
        return ""
    # Drop LaTeX commands/macros but keep their argument text when possible.
    # (Converted/OCR TeX often wraps headings like "\\subsection*{Chapter 3 ...}".)
    s2 = s
    for _ in range(3):
        ns = _LATEX_CMD_WITH_ARG_RE.sub(r" \1 ", s2)
        if ns == s2:
            break
        s2 = ns
    s2 = _LATEX_CMD_RE.sub(" ", s2)

    s2 = s2.lower()
    s2 = _PUNCT_RE.sub(" ", s2)
    s2 = " ".join(s2.split())

    # If CJK exists, insert spaces between characters to improve token-set matching.
    if re.search(r"[\u4e00-\u9fff]", s2):
        s2 = re.sub(r"([\u4e00-\u9fff])", r" \1 ", s2)
        s2 = " ".join(s2.split())
    return s2


def token_set(s: str) -> Set[str]:
    toks = tokenize_simple(s)
    return set(toks)


def token_jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity on simple word tokens."""
    sa = token_set(a)
    sb = token_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(max(1, union))


def title_similarity(a: str, b: str) -> float:
    """Composite title similarity for chapter mapping.

    Combines token_set_ratio (or SequenceMatcher fallback) with token Jaccard.
    """
    a2 = normalize_title_robust(a)
    b2 = normalize_title_robust(b)
    if not a2 or not b2:
        return 0.0
    r = ratio(a2, b2)
    j = token_jaccard(a2, b2)
    # Weight ratio more heavily but require token overlap to avoid false positives.
    return float(0.75 * r + 0.25 * j)

def ratio(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    if fuzz is not None:
        # token_set_ratio is whitespace-token based; spacing CJK chars above helps.
        return float(fuzz.token_set_ratio(a, b)) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def partial_ratio(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    if fuzz is not None:
        return float(fuzz.partial_ratio(a, b)) / 100.0
    # fallback: check small windows only
    if len(b) < len(a):
        a, b = b, a
    if not a:
        return 0.0
    # sample windows
    w = len(a)
    best = 0.0
    step = max(1, w//4)
    for i in range(0, max(1, len(b)-w+1), step):
        cand = b[i:i+w]
        best = max(best, difflib.SequenceMatcher(None, a, cand).ratio())
        if best >= 0.99:
            break
    return best

def find_all(haystack: str, needle: str, limit: int=50) -> List[int]:
    out = []
    if not needle:
        return out
    start = 0
    while len(out) < limit:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        out.append(idx)
        start = idx + max(1, len(needle)//2)
    return out

def tokenize_simple(s: str) -> List[str]:
    s = normalize_title_robust(s)
    if not s:
        return []
    toks: List[str] = []
    for t in s.split():
        if re.fullmatch(r"[\u4e00-\u9fff]", t):
            toks.append(t)
        elif len(t) >= 3:
            toks.append(t)
    return toks
