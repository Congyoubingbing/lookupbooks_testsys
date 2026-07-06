from __future__ import annotations

"""Monotone alignment between PDF TOC chapters and TeX/TXT headings.

We need a mapping pdf_idx -> tex_idx, but real-world books often have:
- extra PDF TOC entries that are not present in TeX headings (foreword, appendices)
- extra TeX headings that are not in the TOC

So the alignment must allow skipping on BOTH sides.
"""

from dataclasses import dataclass
import re
from typing import Dict, List

from .similarity import title_similarity


@dataclass
class Chapter:
    no: int
    title: str


def align_chapter_sequences(pdf_chapters: List[Chapter], tex_chapters: List[Chapter]) -> Dict[int, int]:
    """Return a monotone mapping from pdf chapter indices to tex chapter indices.

    Uses dynamic programming with three actions:
    - match: consume one PDF chapter and one TeX heading
    - skip_pdf: consume one PDF chapter without a match
    - skip_tex: consume one TeX heading without a match

    The returned mapping is sparse: only matched PDF chapters appear.
    """

    n = len(pdf_chapters)
    m = len(tex_chapters)
    INF = 10**9

    # dp[i][j] = best cost aligning first i pdf items with first j tex items
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    # Penalties: tuned to prefer matches but allow skipping.
    SKIP_PDF = 2.0
    SKIP_TEX = 2.0

    _ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
    _DIGIT_RE = re.compile(r"^\d+$")
    _CJK_NUM_RE = re.compile(r"^[一二三四五六七八九十百千〇零]+$")
    _EN_CHAP_RE = re.compile(r"^(?:CHAPTER|Chapter|Chap\.?|CHAP\.?)\s*(\d{1,3})\b")
    _ZH_CHAP_RE = re.compile(r"^第\s*([0-9０-９]+|[零〇○一二两三四五六七八九十百千]{1,8})\s*章\b")

    def _is_pure_number_title(t: str) -> bool:
        s = (t or "").strip()
        if not s:
            return False
        s = re.sub(r"[\s\-–—·\.\:;,_()\[\]{}]+", "", s)
        if not s:
            return False
        return bool(_DIGIT_RE.match(s) or _ROMAN_RE.match(s) or _CJK_NUM_RE.match(s))

    def _is_numbered_chapter_heading(t: str) -> bool:
        s = (t or "").strip()
        if not s:
            return False
        if _EN_CHAP_RE.match(s):
            return True
        if _ZH_CHAP_RE.match(s):
            return True
        return _is_pure_number_title(s)

    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur >= INF:
                continue

            # Skip a PDF TOC entry
            if i < n and cur + SKIP_PDF < dp[i + 1][j]:
                dp[i + 1][j] = cur + SKIP_PDF
                bt[i + 1][j] = (i, j, "skip_pdf")

            # Skip a TeX heading
            if j < m and cur + SKIP_TEX < dp[i][j + 1]:
                dp[i][j + 1] = cur + SKIP_TEX
                bt[i][j + 1] = (i, j, "skip_tex")

            # Match
            if i < n and j < m:
                sim = title_similarity(pdf_chapters[i].title, tex_chapters[j].title)
                # inverse similarity as cost
                cost = (1.0 - sim) * 10.0
                # soft number mismatch penalty
                if pdf_chapters[i].no and tex_chapters[j].no and pdf_chapters[i].no != tex_chapters[j].no:
                    cost += 1.5

                # Special-case: TeX heading title is only a number/roman numeral and chapter numbers match.
                # In that case, title similarity can be misleadingly low; allow the match with lower cost.
                if pdf_chapters[i].no and tex_chapters[j].no and pdf_chapters[i].no == tex_chapters[j].no and _is_numbered_chapter_heading(tex_chapters[j].title):
                    # Make numbered chapter-line matches decisively cheaper than skip+skip.
                    # This avoids dropping mappings for "CHAPTER N" / "第N章" headings whose
                    # lexical similarity to TOC titles is naturally very low.
                    cost = min(cost * 0.10, 0.5)
                else:
                    # Extra guard: if similarity is extremely low, discourage matching.
                    if sim < 0.25:
                        cost += 4.0
                if cur + cost < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = cur + cost
                    bt[i + 1][j + 1] = (i, j, "match")

    # best end among dp[n][*]
    # Tie-break toward larger j (i.e., consuming/matching more TeX headings), otherwise
    # the DP can prefer an equally-cheap path that skips the final numbered chapter heading.
    best_j = min(range(m + 1), key=lambda jj: (dp[n][jj], -jj))
    mapping: Dict[int, int] = {}

    i, j = n, best_j
    while i > 0 or j > 0:
        prev = bt[i][j]
        if prev is None:
            break
        pi, pj, action = prev
        if action == "match":
            mapping[pi] = pj
        i, j = pi, pj

    return mapping
