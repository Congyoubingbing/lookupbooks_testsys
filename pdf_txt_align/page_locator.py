from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging
import math
import re
import statistics

from .pdf_units import PDFUnitStore
from .llm_calls import vl_read_page_label, vl_verify_chapter_start
from .utils import dump_json, load_json, roman_to_int

PAGE_LOC_CACHE_VERSION = 7


def _spearman_corr(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation with average-rank ties."""
    if not xs or not ys or len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    n = len(xs)

    def rank(data: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: data[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and data[order[j + 1]] == data[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return float(cov / math.sqrt(vx * vy))


@dataclass
class PageLocResult:
    printed_page: str
    unit_idx: int
    conf: float
    method: str


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(str(s).strip())
    except Exception:
        return None



def _normalize_label_str(label: str) -> Optional[str]:
    """Normalize raw label strings from VL into a strict label token.

    Accepts arabic digits or roman numerals; strips wrappers like 'Page 12' -> '12'.
    """
    if not label:
        return None
    s = str(label).strip()
    if not s:
        return None
    m = re.search(r"\b([0-9]{1,4})\b", s)
    if m:
        v = int(m.group(1))
        if v <= 0:
            return None
        return str(v)
    m2 = re.search(r"\b([ivxlcdm]{1,8})\b", s.lower())
    if m2:
        return m2.group(1).lower()
    return None

def _parse_page_label_from_text_layer(text: str, *, region: str) -> Optional[str]:
    """Best-effort parse of a printed page label from text layer crops.

    We ONLY accept a line that is *entirely* an arabic number or roman numeral (optionally wrapped by
    simple punctuation like "- 12 -" or "(xii)".

    The text-layer crop can still contain stray numeric-only lines (figure ticks, equation numbers).
    To reduce false positives we enforce:

    - Inspect only a small window of lines near the header/footer edge (region-dependent).
    - The candidate line must be very close to the crop edge (top: first lines; bottom: last lines).
    - Within the inspected window, there must be exactly ONE numeric-only candidate.
    """
    if not text:
        return None
    region = (region or "").strip().lower()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None

    # The crop already constrains header/footer region; still inspect only a few lines.
    if region == "top":
        win = lines[:16]
        edge_max_idx = 3  # must appear within first few lines
    else:
        win = lines[-16:]
        edge_max_idx = 3  # must appear within last few lines (relative to win)

    cands: list[tuple[str, int]] = []
    for idx, ln in enumerate(win):
        m = re.fullmatch(r"[\(\[\{<\-\s]*([0-9]{1,4})[\)\]\}>\-\s]*", ln)
        if m:
            cands.append((m.group(1), idx))
            continue
        m2 = re.fullmatch(r"[\(\[\{<\-\s]*([ivxlcdm]{1,8})[\)\]\}>\-\s]*", ln.lower())
        if m2:
            cands.append((m2.group(1).lower(), idx))

    if not cands:
        return None

    # Reject if multiple numeric-only candidates appear (high chance of chart ticks / equation numbers).
    uniq_vals = list(dict.fromkeys([v for v, _i in cands if v]))
    if len(uniq_vals) != 1:
        return None
    val = uniq_vals[0].strip()
    if not val:
        return None

    # Enforce edge proximity: page labels are typically isolated near the extreme header/footer edge.
    cand_idx = [i for v, i in cands if v == val]
    if not cand_idx:
        return None
    idx0 = cand_idx[0]
    if region == "top":
        if idx0 > edge_max_idx:
            return None
    else:
        if idx0 < max(0, len(win) - 1 - edge_max_idx):
            return None

    # reject 0 / negative
    if val.isdigit():
        try:
            iv = int(val)
            if iv <= 0:
                return None
        except Exception:
            return None
    return val

def _toc_likeness(text: str) -> float:
    """Cheap TOC-likeness score from text layer to reject TOC/index pages."""
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


class PageLocator:
    """Map TOC printed pages to PDF unit indices.

    Artifacts:
      - _cache/page_locator.json
    """

    def __init__(self, store: PDFUnitStore, session_vl, cfg, cache_dir: Path, logger: logging.Logger):
        self.store = store
        self.session_vl = session_vl
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.logger = logger


    def _build_label_crops(self, img_full, crop_policies: List[str]) -> Tuple[List[Any], List[str]]:
        """Build crops for header/footer page-label reading.

        Returns (crops, crop_names) aligned with the list given to the VL model.
        We expand '*_outer' into left+right to handle even/odd outer-side variation.
        """
        w, h = img_full.size
        y0_footer = int(h * float(getattr(self.cfg.pdf, "page_label_crop_y0_ratio", 0.82) or 0.82))
        y1_header = int(h * float(getattr(self.cfg.pdf, "page_label_crop_top_y1_ratio", 0.22) or 0.22))
        y0_footer = max(0, min(h - 2, y0_footer))
        y1_header = max(2, min(h, y1_header))

        def crop_box(pol: str):
            pol = (pol or "").lower().strip()
            if pol.startswith("top"):
                y0, y1 = 0, y1_header
                base = "top"
            else:
                y0, y1 = y0_footer, h
                base = "bottom"

            # horizontal slices
            if pol.endswith("_left"):
                x0, x1 = 0, int(w * 0.50)
            elif pol.endswith("_right"):
                x0, x1 = int(w * 0.50), w
            elif pol.endswith("_center"):
                x0, x1 = int(w * 0.25), int(w * 0.75)
            else:
                x0, x1 = int(w * 0.10), int(w * 0.90)

            x0 = max(0, min(w - 2, x0))
            x1 = max(x0 + 2, min(w, x1))
            y0 = max(0, min(h - 2, y0))
            y1 = max(y0 + 2, min(h, y1))
            return (x0, y0, x1, y1), f"{base}:{pol}"

        crops: List[Any] = []
        names: List[str] = []

        for pol in crop_policies or []:
            pol_s = str(pol or "").lower().strip()
            # expand outer into both sides
            if pol_s in ("bottom_outer", "top_outer"):
                for side in (pol_s.replace("_outer", "_left"), pol_s.replace("_outer", "_right")):
                    box, name = crop_box(side)
                    try:
                        crops.append(img_full.crop(box))
                    except Exception:
                        crops.append(img_full)
                    names.append(name)
                continue

            box, name = crop_box(pol_s)
            try:
                crops.append(img_full.crop(box))
            except Exception:
                crops.append(img_full)
            names.append(name)

        # final fallback if nothing configured
        if not crops:
            box, name = crop_box("bottom_center")
            try:
                crops = [img_full.crop(box)]
            except Exception:
                crops = [img_full]
            names = [name]

        return crops, names

    def _read_label_on_unit(self, unit_idx: int, *, allow_vl: bool = True) -> Tuple[Optional[str], float, str]:
        """Read printed page label on a given unit. Returns (label, conf, method)."""

        # (A) text-layer reading (fast): try header then footer by default
        regions = list(getattr(self.cfg.pdf, "page_label_text_regions", []) or [])
        if not regions:
            regions = ["top", "bottom"]

        for reg in regions:
            try:
                if reg == "bottom":
                    t = self.store.extract_unit_text(
                        self.store.unit_ref(unit_idx),
                        region="bottom",
                        y0_ratio=float(getattr(self.cfg.pdf, "page_label_crop_y0_ratio", 0.82) or 0.82),
                    )
                else:
                    # top region uses a fixed 25% strip in PDFUnitStore; keep text-layer parse strict.
                    t = self.store.extract_unit_text(self.store.unit_ref(unit_idx), region="top")
            except Exception:
                t = ""

            label = _parse_page_label_from_text_layer(t or "", region=reg)
            if not label:
                continue

            # Reject TOC/Index-like pages even if a label is present.
            try:
                reject_score = float(getattr(self.cfg.pdf, "page_label_reject_toc_score", 0.65) or 0.65)
                if reject_score > 0:
                    full_t = self.store.extract_unit_text(self.store.unit_ref(unit_idx), region="full")
                    if _toc_likeness(full_t or "") >= reject_score:
                        label = None
            except Exception:
                pass

            if label:
                return label, 0.70, f"text_layer:{reg}"

        if not allow_vl or not bool(getattr(self.cfg.pdf, "enable_page_label_reading", True)):
            return None, 0.0, "none"

        # (B) Vision reading: try multiple header+footer crops
        crop_policies = list(getattr(self.cfg.pdf, "page_label_crop_policies", []) or [])
        # Reorder crop policies by language/layout hints:
        # - English books often place printed page numbers in outer headers.
        # - Chinese books often place printed page numbers in outer footers.
        hint = str(getattr(self.cfg.pdf, "page_label_language_hint", "") or "").lower()
        if hint:
            def _is_bottom(p: str) -> bool:
                return str(p).lower().startswith("bottom")
            def _is_top(p: str) -> bool:
                return str(p).lower().startswith("top")
            if ("zh" in hint) or ("cn" in hint) or ("cjk" in hint) or ("中文" in hint):
                crop_policies = sorted(crop_policies, key=lambda p: (0 if _is_bottom(p) else 1))
            elif ("en" in hint) or ("eng" in hint) or ("latin" in hint):
                crop_policies = sorted(crop_policies, key=lambda p: (0 if _is_top(p) else 1))
        if not crop_policies:
            crop_policies = ["top_outer", "top_left", "top_right", "bottom_outer", "bottom_center", "bottom_right", "bottom_left"]

        img_full = self.store.render_unit(self.store.unit_ref(unit_idx), dpi=self.store.dpi_low, region="full")
        crops, crop_names = self._build_label_crops(img_full, crop_policies)

        try:
            out = vl_read_page_label(
                self.session_vl,
                crops,
                model=str(getattr(self.cfg.models, "vision_model", "qwen3-vl-plus")),
                enable_thinking=bool(getattr(self.cfg.models, "vision_enable_thinking", False)),
                crop_policy="|".join([str(p) for p in crop_names]),
            )
        except TypeError:
            out = vl_read_page_label(
                self.session_vl,
                crops,
                model=str(getattr(self.cfg.models, "vision_model", "qwen3-vl-plus")),
                enable_thinking=bool(getattr(self.cfg.models, "vision_enable_thinking", False)),
            )

        lab_raw = str((out or {}).get("label", "") or "").strip()
        lab = _normalize_label_str(lab_raw)
        conf = float((out or {}).get("conf", 0.0) or 0.0)
        crop_index = (out or {}).get("crop_index", None)

        if lab and conf >= float(getattr(self.cfg.pdf, "page_label_min_conf", 0.65) or 0.65):
            pol = None
            try:
                if crop_index is not None:
                    ci = int(crop_index)
                    if 0 <= ci < len(crop_names):
                        pol = crop_names[ci]
            except Exception:
                pol = None
            return str(lab).strip(), float(conf), f"vision:{pol or 'unknown'}"

        return None, float(conf), "vision_low_conf"

    def _collect_page_label_samples(self, toc_end_unit: int) -> Tuple[List[Tuple[int, int, float, str]], int]:
        """Collect (printed_page_int, unit_idx, conf, method) samples after TOC.

        Strategy:
        - Pass A: text-layer only (fast)
        - If points are insufficient OR regression confidence is low, run Pass B with VL crops.
        """
        enable = bool(getattr(self.cfg.pdf, "enable_page_label_regression", True))
        if not enable:
            return [], 0

        scan_units = int(getattr(self.cfg.pdf, "page_label_sample_scan_units", 260) or 260)
        stride = int(getattr(self.cfg.pdf, "page_label_sample_stride", 4) or 4)
        target = int(getattr(self.cfg.pdf, "page_label_samples_target", 12) or 12)
        max_vl = int(getattr(self.cfg.pdf, "page_label_max_vl_calls", 24) or 24)
        min_pts = int(getattr(self.cfg.pdf, "page_label_fit_min_points", 4) or 4)

        start = max(0, int(toc_end_unit) + 1)
        end = min(self.store.unit_count - 1, start + scan_units)

        points: List[Tuple[int, int, float, str]] = []
        seen_pages: set[int] = set()
        seen_units: set[int] = set()
        vl_calls = 0

        # pass A: text-layer only
        for ui in range(start, end + 1, stride):
            lab, conf, method = self._read_label_on_unit(ui, allow_vl=False)
            if not lab:
                continue
            p = _safe_int(lab)
            if p is None or p <= 0:
                continue
            if p in seen_pages:
                continue
            points.append((p, ui, float(conf), method))
            seen_pages.add(p)
            seen_units.add(ui)
            if len(points) >= target:
                break

        # decide whether to run VL sampling even if we have enough points
        need_vl = len(points) < min_pts
        if not need_vl and max_vl > 0:
            try:
                model_a = self._fit_page_label_model(points)
                min_conf_use = float(getattr(self.cfg.pdf, "page_label_min_conf_use", 0.72) or 0.72)
                min_spear = float(getattr(self.cfg.pdf, "page_label_min_spearman", 0.75) or 0.75)
                if (model_a is None) or (float(model_a.get("conf", 0.0) or 0.0) < min_conf_use) or (float(model_a.get("spearman", 0.0) or 0.0) < min_spear):
                    need_vl = True
            except Exception:
                need_vl = True

        # pass B: VL sampling
        if need_vl and max_vl > 0:
            for ui in range(start, end + 1, stride):
                if vl_calls >= max_vl:
                    break
                # skip units already sampled by text-layer success
                if ui in seen_units:
                    continue

                lab, conf, method = self._read_label_on_unit(ui, allow_vl=True)
                if method.startswith("text_layer"):
                    continue
                vl_calls += 1
                if not lab:
                    continue
                p = _safe_int(lab)
                if p is None or p <= 0:
                    continue
                if p in seen_pages:
                    continue
                points.append((p, ui, float(conf), method))
                seen_pages.add(p)
                if len(points) >= target:
                    break


        # Final cleaning: if regression quality remains very low, prefer VL-derived samples over text-layer samples.
        try:
            min_conf_use = float(getattr(self.cfg.pdf, "page_label_min_conf_use", 0.72) or 0.72)
            min_spear = float(getattr(self.cfg.pdf, "page_label_min_spearman", 0.75) or 0.75)
            min_pts2 = int(getattr(self.cfg.pdf, "page_label_fit_min_points", 4) or 4)

            model_all = self._fit_page_label_model(points) if points else None
            if model_all:
                conf_all = float(model_all.get("conf", 0.0) or 0.0)
                sp_all = float(model_all.get("spearman", 0.0) or 0.0)
                # NOTE: VL-derived page-label samples are recorded with method prefix "vision:".
                has_vl = any((m or "").startswith("vision:") for _p, _u, _c, m in points)
                has_text = any((m or "").startswith("text_layer") for _p, _u, _c, m in points)

                # If we have VL points and the fit is poor, drop text-layer points and refit.
                if has_vl and has_text and (conf_all < min_conf_use or sp_all < min_spear or conf_all <= 0.25):
                    pts_vl = [pt for pt in points if not (pt[3] or "").startswith("text_layer")]
                    model_vl = self._fit_page_label_model(pts_vl) if len(pts_vl) >= min_pts2 else None
                    if model_vl:
                        conf_vl = float(model_vl.get("conf", 0.0) or 0.0)
                        sp_vl = float(model_vl.get("spearman", 0.0) or 0.0)
                        if (conf_vl > conf_all) or (sp_vl > sp_all and conf_vl >= max(0.1, conf_all)):
                            points = pts_vl
                            model_all = model_vl
                            conf_all = conf_vl
                            sp_all = sp_vl

                # If still poor, remove extreme residual outliers and refit once.
                if model_all and len(points) >= min_pts2 and (conf_all < min_conf_use or sp_all < min_spear or conf_all <= 0.25):
                    a = float(model_all.get("a", 1.0) or 1.0)
                    b = float(model_all.get("b", 0.0) or 0.0)
                    max_abs_err = float(getattr(self.cfg.pdf, "page_label_fit_max_abs_err", 3.0) or 3.0)
                    thr = max(8.0, 3.0 * max_abs_err)
                    pts_f = [pt for pt in points if abs(float(pt[1]) - (a * float(pt[0]) + b)) <= thr]
                    if len(pts_f) >= min_pts2 and len(pts_f) < len(points):
                        model_f = self._fit_page_label_model(pts_f)
                        if model_f and float(model_f.get("conf", 0.0) or 0.0) >= conf_all:
                            points = pts_f
        except Exception:
            pass

        return points, vl_calls

    def _fit_page_label_model(self, points: List[Tuple[int, int, float, str]]) -> Optional[Dict[str, Any]]:
        """Fit robust linear model unit ≈ a*page + b."""
        min_pts = int(getattr(self.cfg.pdf, "page_label_fit_min_points", 4) or 4)
        if len(points) < min_pts:
            return None

        pts_all = sorted([(p, u) for p, u, _c, _m in points], key=lambda t: t[0])

        max_abs = int(getattr(self.cfg.pdf, "page_label_max_printed_page_abs", 1200) or 1200)
        buf = int(getattr(self.cfg.pdf, "page_label_max_printed_page_buffer", 100) or 100)
        max_by_units = int(self.store.unit_count * 2 + buf)
        max_p = max(50, min(max_abs, max_by_units))
        pts = [(p, u) for (p, u) in pts_all if (isinstance(p, int) and 0 < p <= max_p)]
        dropped = max(0, len(pts_all) - len(pts))
        if len(pts) < min_pts:
            return None

        spearman = _spearman_corr([float(p) for p, _u in pts], [float(u) for _p, u in pts])

        slopes: List[float] = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                p1, u1 = pts[i]
                p2, u2 = pts[j]
                dp = p2 - p1
                if dp == 0:
                    continue
                slopes.append((u2 - u1) / float(dp))
        a = float(statistics.median(slopes)) if slopes else 1.0

        # Allow two-up layouts (unit-per-printed-page ≈ 2) while still rejecting extreme slopes.
        layout = str(getattr(getattr(self.store, 'layout_info', None), 'layout', '') or '').lower()
        if layout == 'two_up_lr':
            if not ((0.6 <= a <= 1.4) or (1.6 <= a <= 2.4)):
                a = 2.0
        else:
            if not (0.6 <= a <= 1.4):
                a = 1.0

        bs = [u - a * p for p, u in pts]
        b = float(statistics.median(bs)) if bs else 0.0

        residuals = [abs(u - (a * p + b)) for p, u in pts]
        mad = float(statistics.median(residuals)) if residuals else 999.0
        max_abs_err = float(getattr(self.cfg.pdf, "page_label_fit_max_abs_err", 3.0) or 3.0)
        conf = max(0.0, min(0.99, 1.0 - (mad / max(1e-6, max_abs_err))))

        min_spear = float(getattr(self.cfg.pdf, "page_label_min_spearman", 0.75) or 0.75)
        if spearman < min_spear:
            conf = 0.0

        return {
            "a": a,
            "b": b,
            "mad": mad,
            "conf": conf,
            "points": [{"page": p, "unit": u} for p, u in pts],
            "dropped_points": dropped,
            "spearman": float(spearman),
        }

    def locate_page1_after_toc(self, toc_end_unit: int) -> Optional[PageLocResult]:
        """Find printed page '1' in a window after TOC.

        NOTE: In practice, TOC detection can be wrong (e.g., picking a chapter-local mini-TOC deep in the book).
        To avoid scanning a completely wrong late window, we first perform a guarded early scan near the front
        matter, then fall back to the TOC-adjacent window.
        """

        min_conf = float(getattr(self.cfg.pdf, "page_label_min_conf", 0.65) or 0.65)
        reject_toc = float(getattr(self.cfg.pdf, "page1_reject_toc_score", 0.80) or 0.80)
        trust_early_max = int(getattr(self.cfg.pdf, "page1_trust_early_max_unit", 160) or 160)

        def _scan(lo: int, hi: int, *, stride: int = 1, require_sequence: bool = False) -> Optional[PageLocResult]:
            lo = max(0, int(lo))
            hi = min(self.store.unit_count - 1, int(hi))
            if hi < lo:
                return None

            best: Optional[PageLocResult] = None
            for ui in range(lo, hi + 1, max(1, int(stride))):
                # Avoid TOC-like pages which frequently contain stray digits near the footer.
                try:
                    full_text = self.store.extract_unit_text(self.store.unit_ref(ui), region="full")
                except Exception:
                    full_text = ""
                if full_text and _toc_likeness(full_text) >= reject_toc:
                    continue

                lab, conf, method = self._read_label_on_unit(ui, allow_vl=True)
                if not lab:
                    continue
                if str(lab).strip() != "1" or float(conf) < min_conf:
                    continue

                # Optional sequence check ("2" or "3" soon after) to reduce false positives.
                if require_sequence:
                    ok = False
                    max_ahead = int(getattr(self.cfg.pdf, "page1_sequence_verify_ahead", 6) or 6)
                    required = int(getattr(self.cfg.pdf, "page1_sequence_verify_required", 1) or 1)
                    hits = 0
                    for off in range(1, max_ahead + 1):
                        nxt = ui + off
                        if nxt > self.store.unit_count - 1:
                            break
                        lab2, conf2, _m2 = self._read_label_on_unit(nxt, allow_vl=False)
                        if not lab2:
                            continue
                        if str(lab2).strip() == str(off + 1) and float(conf2) >= min_conf:
                            hits += 1
                            if hits >= required:
                                ok = True
                                break
                    if not ok:
                        continue

                cur = PageLocResult("1", int(ui), float(conf), method)
                if best is None or float(cur.conf) > float(best.conf):
                    best = cur
                # Early exit on very high confidence.
                if float(conf) >= 0.90:
                    return best
            return best

        # Pass 0: early scan near the front matter.
        # In many books, TOC detection can be wrong; also, early pages may omit the "1" label while later
        # arabic labels (e.g., 5, 7, 10) exist. We therefore collect small arabic labels and back-calculate
        # the unit index for printed page 1 when possible.
        early_units = int(getattr(self.cfg.pdf, "page1_scan_first_units", 120) or 120)
        early_stride = int(getattr(self.cfg.pdf, "page1_scan_stride", 1) or 1)
        backcalc_max_p = int(getattr(self.cfg.pdf, "page1_backcalc_max_printed_page", 50) or 50)
        backcalc_min_pts = int(getattr(self.cfg.pdf, "page1_backcalc_min_points", 2) or 2)
        backcalc_tol = int(getattr(self.cfg.pdf, "page1_backcalc_offset_tol_units", 3) or 3)
        if early_units > 0:
            hi0 = min(self.store.unit_count - 1, int(early_units))
            self.logger.info(f"[PAGE] scanning for printed page '1' in early units [0,{hi0}]")

            best: Optional[PageLocResult] = None
            offsets: List[int] = []
            for ui in range(0, hi0 + 1, max(1, int(early_stride))):
                # Avoid TOC-like pages which frequently contain stray digits near the footer.
                try:
                    full_text = self.store.extract_unit_text(self.store.unit_ref(ui), region="full")
                except Exception:
                    full_text = ""
                if full_text and _toc_likeness(full_text) >= reject_toc:
                    continue

                lab, conf, method = self._read_label_on_unit(ui, allow_vl=True)
                if not lab:
                    continue

                s = str(lab).strip()
                p_i = _safe_int(s) if re.fullmatch(r"\d{1,4}", s) else None
                if p_i is not None and 1 <= int(p_i) <= int(backcalc_max_p) and float(conf) >= min_conf:
                    offsets.append(int(ui) - (int(p_i) - 1))

                if s == "1" and float(conf) >= min_conf:
                    cur = PageLocResult("1", int(ui), float(conf), method)
                    if best is None or float(cur.conf) > float(best.conf):
                        best = cur
                    if float(conf) >= 0.90:
                        return best

            if best is not None:
                return best

            # Back-calc page-1 unit from small arabic labels if they are self-consistent.
            if len(offsets) >= int(backcalc_min_pts):
                med = int(statistics.median(offsets))
                spread = int(max(offsets) - min(offsets)) if offsets else 999
                if spread <= int(backcalc_tol) and 0 <= med <= (self.store.unit_count - 1):
                    return PageLocResult("1", int(med), 0.25, "page1_backcalc_early")


        # If TOC end is very late relative to the front-matter scan window, it is frequently a false TOC hit
        # (e.g., chapter-local mini-TOC). Dense scanning for printed page '1' after such a toc_end is both
        # slow and unreliable. In that case, skip the TOC-adjacent scan and rely on regression-based inference.
        try:
            if int(toc_end_unit) > int(max(0, int(early_units)) * 0.6):
                self.logger.info(f"[PAGE] toc_end_unit={int(toc_end_unit)} too late vs early_units={int(early_units)}; skip TOC-window page1 scan")
                return None
        except Exception:
            pass

        # Pass 1: scan after TOC end (original behavior).
        win = int(getattr(self.cfg.pdf, "toc_scan_max_units_pass1", 260) or 260)
        start = max(0, int(toc_end_unit) + 1)
        end = min(self.store.unit_count - 1, start + win)
        self.logger.info(f"[PAGE] scanning for printed page '1' in units [{start},{end}]")

        # If this window is far from the beginning, require a minimal sequence check to reduce false positives.
        require_seq = bool(start > trust_early_max)
        res1 = _scan(start, end, stride=1, require_sequence=require_seq)
        if res1 is not None:
            return res1

        self.logger.warning("[PAGE] printed page 1 not found")
        return None

    def _search_chapter_start_near(self, unit_guess: int, chapter_no: int, chapter_title: str, *, floor_unit: int = 0) -> Optional[PageLocResult]:
        """Verify chapter opening near a unit guess by VL score (reject TOC-like pages)."""
        win = int(getattr(self.cfg.pdf, "chapter_start_search_window_units", 18) or 18)
        min_score = float(getattr(self.cfg.pdf, "chapter_start_min_score", 0.75) or 0.75)
        reject_toc = float(getattr(self.cfg.pdf, "chapter_start_reject_toc_score", 0.72) or 0.72)
        floor_unit = max(0, int(floor_unit))
        lo = max(floor_unit, int(unit_guess) - win)
        hi = min(self.store.unit_count - 1, int(unit_guess) + win)

        best_ui = None
        best_score = 0.0
        for ui in range(lo, hi + 1):
            try:
                full_text = self.store.extract_unit_text(self.store.unit_ref(ui), region="full")
            except Exception:
                full_text = ""
            if full_text and _toc_likeness(full_text) >= reject_toc:
                continue

            img = self.store.render_unit(self.store.unit_ref(ui), dpi=self.store.dpi_low, region="body")
            out = vl_verify_chapter_start(
                self.session_vl,
                img,
                int(chapter_no or 0),
                str(chapter_title or ""),
                model=str(getattr(self.cfg.models, "vision_model", "qwen3-vl-plus")),
                enable_thinking=bool(getattr(self.cfg.models, "vision_enable_thinking", False)),
            )
            score = float((out or {}).get("score", 0.0) or 0.0)
            is_start = bool((out or {}).get("is_start", True))
            if (not is_start) and score < 0.90:
                continue
            if score > best_score:
                best_score = score
                best_ui = int(ui)

        if best_ui is not None and best_score >= min_score:
            return PageLocResult("", int(best_ui), float(best_score), "chapter_start_verify")
        return None

    def _search_chapter_start_forward(
        self,
        start_unit: int,
        end_unit: int,
        *,
        chapter_no: int,
        chapter_title: str,
        floor_unit: int = 0,
    ) -> Optional[PageLocResult]:
        """Search chapter start in a forward-only window [start_unit, end_unit]."""
        floor_unit = max(0, int(floor_unit))
        start_unit = max(floor_unit, int(start_unit))
        end_unit = min(self.store.unit_count - 1, int(end_unit))
        if end_unit < start_unit:
            return None

        min_score = float(getattr(self.cfg.pdf, "chapter_start_min_score", 0.75) or 0.75)
        reject_toc = float(getattr(self.cfg.pdf, "chapter_start_reject_toc_score", 0.72) or 0.72)

        best_ui = None
        best_score = 0.0
        for ui in range(start_unit, end_unit + 1):
            try:
                full_text = self.store.extract_unit_text(self.store.unit_ref(ui), region="full")
            except Exception:
                full_text = ""
            if full_text and _toc_likeness(full_text) >= reject_toc:
                continue
            img = self.store.render_unit(self.store.unit_ref(ui), dpi=self.store.dpi_low, region="body")
            out = vl_verify_chapter_start(
                self.session_vl,
                img,
                int(chapter_no or 0),
                str(chapter_title or ""),
                model=str(getattr(self.cfg.models, "vision_model", "qwen3-vl-plus")),
                enable_thinking=bool(getattr(self.cfg.models, "vision_enable_thinking", False)),
            )
            score = float((out or {}).get("score", 0.0) or 0.0)
            is_start = bool((out or {}).get("is_start", True))
            if (not is_start) and score < 0.90:
                continue
            if score > best_score:
                best_score = score
                best_ui = int(ui)

        if best_ui is not None and best_score >= min_score:
            return PageLocResult("", int(best_ui), float(best_score), "chapter_start_verify_forward")
        return None

    def map_chapter_start_units(
        self,
        chapters: List[Dict[str, Any]],
        *,
        toc_end_unit: int,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Map each chapter's printed page to a PDF unit index."""
        cache_path = self.cache_dir / "page_locator.json"
        cached = load_json(cache_path)
        if cached and isinstance(cached, dict) and cached.get("chapters"):
            try:
                if int(cached.get("_cache_version", 0) or 0) == PAGE_LOC_CACHE_VERSION and isinstance(cached.get("chapters"), list):
                    if len(cached.get("chapters") or []) == len(chapters):
                        return cached.get("chapters", []), cached.get("fallbacks", [])
            except Exception:
                pass

        fallbacks: List[str] = []

        # Optional metadata recorded into page_locator.json
        page1_inferred_unit: Optional[int] = None

        page1 = self.locate_page1_after_toc(toc_end_unit)
        page1_unit = page1.unit_idx if page1 else None
        if page1_unit is None:
            fallbacks.append("page1_not_found")

        # If TOC end is after the detected page-1 unit, it is almost certainly a false TOC hit.
        # Clamp the sampling start so regression uses the correct arabic-numbering region.
        effective_toc_end = int(toc_end_unit)
        if page1_unit is not None and int(toc_end_unit) > int(page1_unit):
            fallbacks.append(f"toc_end_after_page1_clamped:{toc_end_unit}->{int(page1_unit) - 1}")
            effective_toc_end = max(0, int(page1_unit) - 1)

        samples, vl_calls = self._collect_page_label_samples(effective_toc_end)
        model = self._fit_page_label_model(samples)
        # Guard: if all regression samples are from relatively large printed pages (e.g., >30),
        # even a perfect fit does NOT justify high confidence for inferring printed page "1".
        # Clamp model confidence to avoid false-positive page1 inference.
        if model and samples:
            try:
                min_sample_page = min(int(p) for (p, _u, _c, _m) in samples if p is not None)
                high_thr = int(getattr(self.cfg.pdf, "infer_page1_high_pages_threshold", 30) or 30)
                cap = float(getattr(self.cfg.pdf, "infer_page1_untrusted_conf_cap", 0.7) or 0.7)
                if min_sample_page > int(high_thr):
                    if float(model.get("conf", 0.0) or 0.0) > float(cap):
                        model["conf"] = float(cap)
                    fallbacks.append(f"page1_model_high_pages_untrusted:min_page={min_sample_page}")
            except Exception:
                pass
        if model:
            self.logger.info(
                f"[PAGE] fitted label model a={model['a']:.3f}, b={model['b']:.2f}, mad={model['mad']:.2f}, conf={model['conf']:.2f}"
            )
        else:
            fallbacks.append("page_label_model_insufficient")

        # If we failed to locate printed page "1" reliably, infer it from the fitted model.
        # Many scanned PDFs omit a visible page label on early pages; requiring an explicit '1' detection
        # makes the pipeline waste time scanning wrong windows. We therefore allow *unverified* inference
        # from the regression model, and only do a small local verification when possible.
        if page1_unit is None and model and float(model.get("conf", 0.0) or 0.0) >= float(getattr(self.cfg.pdf, "infer_page1_min_conf", 0.80) or 0.80):
            try:
                u1 = int(round(float(model.get("a", 1.0)) * 1.0 + float(model.get("b", 0.0))))
                infer_max = int(getattr(self.cfg.pdf, "infer_page1_max_unit", 200) or 200)
                infer_max = max(0, min(self.store.unit_count - 1, infer_max))
                # Guard: inferred page1 should be reasonably near the beginning.
                if 0 <= u1 <= infer_max:
                    page1_unit = int(u1)
                    page1_inferred_unit = int(u1)
                    fallbacks = [fb for fb in fallbacks if fb != "page1_not_found"]
                    fallbacks.append("page1_inferred_from_model")

                    # Optional local verification (does not block inference).
                    win = int(getattr(self.cfg.pdf, "infer_page1_verify_window_units", 2) or 2)
                    cand_units = [u for u in range(max(0, u1 - win), min(self.store.unit_count - 1, u1 + win) + 1)]
                    best = None
                    for ui in cand_units:
                        p, c, _method = self._read_label_on_unit(int(ui))
                        if p is not None and str(p).strip() == "1" and c >= float(getattr(self.cfg.pdf, "page_label_min_conf", 0.65) or 0.65):
                            if best is None or c > best[1]:
                                best = (int(ui), float(c))
                    if best is not None:
                        page1_unit = int(best[0])
                        page1_inferred_unit = int(best[0])
                        fallbacks.append("page1_inferred_verified")
            except Exception:
                pass

        # If TOC end is after the (detected or inferred) page-1 unit, it is almost certainly a false TOC hit.
        # Clamp the sampling/search floor so chapter starts are not matched into front-matter/TOC pages.
        if page1_unit is not None and int(toc_end_unit) > int(page1_unit):
            fallbacks.append(f"toc_end_after_page1_clamped:{toc_end_unit}->{int(page1_unit) - 1}")
            effective_toc_end = max(0, int(page1_unit) - 1)

        if (not model) and page1_unit is not None:
            model = {
                "a": 1.0,
                "b": float(page1_unit) - 1.0,
                "mad": 999.0,
                "conf": 0.0,
                "points": [{"page": 1, "unit": page1_unit}],
            }

        min_model_conf = float(getattr(self.cfg.pdf, "page_label_min_conf_use", 0.35) or 0.35)
        min_pts = int(getattr(self.cfg.pdf, "page_label_fit_min_points", 4) or 4)
        model_usable = bool(model) and float(model.get("conf", 0.0) or 0.0) >= min_model_conf and len(model.get("points", []) or []) >= min_pts
        if model and not model_usable:
            fallbacks.append("page_label_model_low_conf")

        floor_unit = max(0, int(effective_toc_end) + 1)

        mapped: List[Dict[str, Any]] = []
        prev_unit: Optional[int] = None
        min_gap = int(getattr(self.cfg.pdf, "min_chapter_unit_gap", 2) or 2)

        for idx, ch in enumerate(chapters):
            no = int(ch.get("no", idx + 1) or (idx + 1))
            title = str(ch.get("title", "") or "").strip()
            pp_raw = str(ch.get("printed_page", "") or "").strip()
            pp = pp_raw

            # printed page numeric (arabic or roman)
            p_int = None
            if pp:
                if re.fullmatch(r"\d{1,4}", pp):
                    p_int = _safe_int(pp)
                else:
                    p_int = roman_to_int(pp)

            # Guard: printed_page==0 (or invalid) is toxic for regression and chapter start inference.
            if p_int is not None and int(p_int) <= 0:
                fallbacks.append(f"invalid_printed_page_ch{no}:{pp_raw}")
                p_int = None
                pp = ""
            elif pp and (p_int is None):
                fallbacks.append(f"invalid_printed_page_ch{no}:{pp_raw}")
                pp = ""

            unit_guess = None
            method = ""
            conf = 0.0

            if p_int is not None and model_usable and model is not None:
                a = float(model.get("a", 1.0) or 1.0)
                b = float(model.get("b", 0.0) or 0.0)
                unit_guess = int(round(a * float(p_int) + b))
                unit_guess = max(0, min(self.store.unit_count - 1, unit_guess))
                method = "page_label_model"
                conf = float(model.get("conf", 0.0) or 0.0)
            elif p_int is not None and page1_unit is not None:
                unit_guess = int(page1_unit + (int(p_int) - 1))
                unit_guess = max(0, min(self.store.unit_count - 1, unit_guess))
                method = "page1_linear"
                conf = 0.20

            found: Optional[PageLocResult] = None

            # verify page label near guess
            if unit_guess is not None and p_int is not None:
                lo = max(floor_unit, unit_guess - 6)
                hi = min(self.store.unit_count - 1, unit_guess + 6)
                best = None
                best_conf = 0.0
                for ui in range(lo, hi + 1):
                    lab, c, _m = self._read_label_on_unit(ui, allow_vl=True)
                    if not lab:
                        continue
                    pi = _safe_int(lab) if re.fullmatch(r"\d{1,4}", str(lab).strip()) else None
                    if pi is not None and pi == p_int and float(c) > best_conf:
                        best = int(ui)
                        best_conf = float(c)
                if best is not None and best_conf >= float(getattr(self.cfg.pdf, "page_label_min_conf", 0.65) or 0.65):
                    found = PageLocResult(str(p_int), int(best), float(best_conf), "page_label")

            if found is None and unit_guess is not None:
                v = self._search_chapter_start_near(unit_guess, no, title, floor_unit=floor_unit)
                if v is not None:
                    found = v

            if found is None and page1_unit is not None and p_int is not None and p_int > 0:
                unit_guess2 = int(page1_unit + (p_int - 1))
                unit_guess2 = max(0, min(self.store.unit_count - 1, unit_guess2))
                v2 = self._search_chapter_start_near(unit_guess2, no, title, floor_unit=floor_unit)
                if v2 is not None:
                    found = v2
                    fallbacks.append(f"chapter_start_verify_used_ch{no}")

            unit_start = int(found.unit_idx) if found is not None else None
            if found is not None:
                conf = float(found.conf)
                method = found.method
            elif unit_guess is not None:
                # v12.4.9: keep a monotone page-label/model guess instead of leaving None.
                # This reduces catastrophic downstream proportional backfill on scan-heavy books.
                unit_start = int(unit_guess)
                method = method or "page_label_guess_unverified"
                conf = max(float(conf), 0.10)
                fallbacks.append(f"chapter_unit_guess_fallback_ch{no}")
            else:
                fallbacks.append(f"chapter_unit_not_found_ch{no}")

            # Monotone repair to prevent cascading drift
            if unit_start is not None and prev_unit is not None and unit_start <= prev_unit:
                if bool(getattr(self.cfg.pdf, "enable_monotone_repair", True)):
                    repaired = min(self.store.unit_count - 1, int(prev_unit) + int(min_gap))
                    fallbacks.append(f"monotone_repair_ch{no}:{unit_start}->{repaired}")
                    lookahead = int(getattr(self.cfg.pdf, "monotone_repair_lookahead_units", 40) or 40)
                    cand = self._search_chapter_start_forward(
                        repaired,
                        min(self.store.unit_count - 1, repaired + lookahead),
                        chapter_no=no,
                        chapter_title=title,
                        floor_unit=floor_unit,
                    )
                    # accept if it passes the same chapter-start threshold
                    min_score = float(getattr(self.cfg.pdf, "chapter_start_min_score", 0.75) or 0.75)
                    if cand is not None and float(cand.conf) >= min_score:
                        unit_start = int(cand.unit_idx)
                        method = cand.method
                        conf = float(cand.conf)
                        fallbacks.append(f"monotone_repair_verified_ch{no}:{repaired}->{unit_start}")
                    else:
                        unit_start = int(repaired)
                else:
                    fallbacks.append(f"non_monotone_ch{no}")

            if unit_start is not None:
                prev_unit = int(unit_start)

            mapped.append({
                "no": no,
                "title": title,
                "printed_page": pp,
                "printed_page_raw": pp_raw,
                "unit_start": unit_start,
                "unit_conf": float(conf),
                "unit_method": method,
            })

        out = {
            "page1": page1.__dict__ if page1 else None,
            "page1_inferred_unit": page1_inferred_unit,
            "samples": [{"page": p, "unit": u, "conf": c, "method": m} for p, u, c, m in samples],
            "model": model,
            "vl_calls": vl_calls,
            "chapters": mapped,
            "fallbacks": fallbacks,
        }
        dump_json(cache_path, out)
        return mapped, fallbacks
