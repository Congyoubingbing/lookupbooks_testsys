from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import math
import statistics

from .utils import ensure_dir

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None


@dataclass
class LayoutInfo:
    layout: str  # "single" or "two_up_lr"
    gutter_x: Optional[int] = None  # in rendered pixels at dpi_low, for two_up_lr


@dataclass
class UnitRef:
    unit_idx: int
    pdf_page_idx: int
    side: Optional[str]  # None | "L" | "R"
    layout: str
    gutter_x: Optional[int] = None


class PDFUnitStore:
    """
    Represents a PDF as a sequence of PAGE-UNITS (original book pages).

    - For normal PDFs: unit_idx == pdf_page_idx, side=None
    - For 2-up scans: each PDF page becomes two units: Left and Right.

    Rendering is on-demand and cached to disk.
    """

    def __init__(
        self,
        pdf_path: Path,
        cache_dir: Path,
        dpi_low: int = 120,
        dpi_high: int = 240,
        two_up_threshold: float = 1.20,
        *,
        force_layout: Optional[str] = None,
        force_gutter_x: Optional[int] = None,
        gutter_detect: bool = True,
        gutter_white_frac_min: float = 0.85,
        gutter_score_min: float = 0.25,
    ):
        if fitz is None:
            raise RuntimeError("PyMuPDF is not installed. Please `pip install pymupdf`.")
        if Image is None:
            raise RuntimeError("Pillow is not installed. Please `pip install pillow`.")
        self.pdf_path = Path(pdf_path)
        self.cache_dir = ensure_dir(cache_dir)
        self.dpi_low = int(dpi_low)
        self.dpi_high = int(dpi_high)
        self.two_up_threshold = float(two_up_threshold)

        self._doc = fitz.open(str(self.pdf_path))
        self.pdf_page_count = self._doc.page_count

        self.layout_info = self._detect_layout(
            force_layout=force_layout,
            force_gutter_x=force_gutter_x,
            gutter_detect=bool(gutter_detect),
            gutter_white_frac_min=float(gutter_white_frac_min),
            gutter_score_min=float(gutter_score_min),
        )

        if self.layout_info.layout == "two_up_lr":
            self.unit_count = self.pdf_page_count * 2
        else:
            self.unit_count = self.pdf_page_count

    def close(self):
        try:
            self._doc.close()
        except Exception:
            pass

    def _detect_layout(
        self,
        *,
        force_layout: Optional[str] = None,
        force_gutter_x: Optional[int] = None,
        gutter_detect: bool = True,
        gutter_white_frac_min: float = 0.85,
        gutter_score_min: float = 0.25,
    ) -> LayoutInfo:
        """
        Detect two-up by aspect ratio and (optionally) center gutter whiteness.

        Why this exists:
        - Some books are true landscape two-up scans (easy: width/height is large).
        - Some landscape PDFs may still be borderline on aspect ratio; optionally use a
          conservative center-gutter (white stripe) detector.

        You can force the layout via config (force_layout), which is the most reliable option
        when auto-detection is ambiguous.
        """

        fl = str(force_layout or "").strip().lower()

        # Explicit override (strongest signal)
        if fl:
            if fl in ("two_up_lr", "two-up_lr", "two_up", "two-up", "double", "doublepage", "2up", "2-up"):
                gut = None
                if force_gutter_x is not None:
                    try:
                        gut = int(force_gutter_x)
                    except Exception:
                        gut = None
                if gut is None:
                    gut = self._estimate_gutter_x(max(0, min(self.pdf_page_count - 1, self.pdf_page_count // 2)))
                return LayoutInfo(layout="two_up_lr", gutter_x=gut)

            if fl in ("single", "one", "1up", "1-up"):
                return LayoutInfo(layout="single", gutter_x=None)

        # sample first N pages (spread a bit)
        sample_n = min(12, self.pdf_page_count)
        idxs = [int(i * (self.pdf_page_count - 1) / max(1, sample_n - 1)) for i in range(sample_n)]
        ratios: List[float] = []
        for pi in idxs:
            page = self._doc.load_page(pi)
            rect = page.rect
            ratios.append((rect.width / rect.height) if rect.height else 0.0)

        # IMPORTANT POLICY:
        # - Portrait PDFs (w/h < ~1.0) are treated as normal single-page books.
        #   This avoids misclassifying true two-column pages (same book page) as 2-up.
        # - Only landscape PDFs are eligible for the center-gutter (white stripe) detector.
        #
        # If you truly have portrait 2-up scans, set `pdf.force_layout: two_up_lr`.
        med_ratio = float(statistics.median(ratios)) if ratios else 0.0
        is_landscape = med_ratio > 1.02
        if not is_landscape:
            return LayoutInfo(layout="single", gutter_x=None)

        # (A) aspect ratio majority vote
        high = [r for r in ratios if r >= self.two_up_threshold]
        if len(high) >= max(2, int(0.6 * len(ratios))):
            gut = self._estimate_gutter_x(idxs[len(idxs) // 2])
            return LayoutInfo(layout="two_up_lr", gutter_x=gut)

        # (B) gutter whiteness detection (only for landscape PDFs)
        if gutter_detect and self.pdf_page_count >= 2:
            try:
                import numpy as np
            except Exception:
                np = None

            if np is not None:
                strong_cols: List[int] = []
                strong = 0

                for pi in idxs:
                    try:
                        img = self.render_pdf_page(pi, dpi=self.dpi_low)
                    except Exception:
                        continue
                    g = np.array(img.convert("L"))
                    if g.ndim != 2:
                        continue
                    h, w = g.shape
                    if w < 80 or h < 80:
                        continue

                    center = w // 2
                    win = max(12, w // 10)
                    lo = max(0, center - win)
                    hi = min(w, center + win)
                    sub = g[:, lo:hi]
                    if sub.size == 0:
                        continue

                    # For each column, compute the fraction of (almost) white pixels.
                    # A true gutter is typically a near-blank stripe across most of the height.
                    white = (sub >= 245).mean(axis=0)  # shape: (hi-lo,)
                    if white.size < 5:
                        continue

                    best_i = int(white.argmax())
                    best_col = int(best_i + lo)
                    best = float(white[best_i])

                    # score against the 2nd-best column (excluding a small neighborhood of the best)
                    white2 = white.copy()
                    nb = 3
                    a = max(0, best_i - nb)
                    b = min(white2.size, best_i + nb + 1)
                    white2[a:b] = -1.0
                    second = float(white2.max()) if white2.size else 0.0
                    score = best - second

                    if best >= float(gutter_white_frac_min) and score >= float(gutter_score_min):
                        strong += 1
                        strong_cols.append(best_col)

                if strong_cols and strong >= max(3, int(0.6 * len(idxs))):
                    gut = int(statistics.median(strong_cols))
                    return LayoutInfo(layout="two_up_lr", gutter_x=gut)

        return LayoutInfo(layout="single", gutter_x=None)

    def _estimate_gutter_x(self, pdf_page_idx: int) -> Optional[int]:
        try:
            img = self.render_pdf_page(pdf_page_idx, dpi=self.dpi_low)
        except Exception:
            return None
        # crude gutter detection: search for the brightest column near center
        try:
            import numpy as np
        except Exception:
            return None

        g = np.array(img.convert("L"))
        h, w = g.shape
        col_mean = g.mean(axis=0)
        center = w // 2
        win = max(10, w // 10)
        lo = max(0, center - win)
        hi = min(w, center + win)
        sub = col_mean[lo:hi]
        if sub.size == 0:
            return center
        best = int(sub.argmax()) + lo
        return best

    def unit_ref(self, unit_idx: int) -> UnitRef:
        if unit_idx < 0 or unit_idx >= self.unit_count:
            raise IndexError(f"unit_idx out of range: {unit_idx}/{self.unit_count}")
        if self.layout_info.layout == "two_up_lr":
            pdf_page_idx = unit_idx // 2
            side = "L" if (unit_idx % 2 == 0) else "R"
            return UnitRef(
                unit_idx=unit_idx,
                pdf_page_idx=pdf_page_idx,
                side=side,
                layout="two_up_lr",
                gutter_x=self.layout_info.gutter_x,
            )
        else:
            return UnitRef(unit_idx=unit_idx, pdf_page_idx=unit_idx, side=None, layout="single", gutter_x=None)

    def render_pdf_page(self, pdf_page_idx: int, dpi: int) -> "Image.Image":
        page = self._doc.load_page(pdf_page_idx)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    def render_unit(self, unit: UnitRef, *, dpi: int, region: str = "full") -> "Image.Image":
        """
        region: full|top|bottom|body
        """
        cache_key = f"unit_{unit.unit_idx:06d}_{region}_{dpi}.png"
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            return Image.open(cache_path)

        base = self.render_pdf_page(unit.pdf_page_idx, dpi=dpi)

        if unit.layout == "two_up_lr":
            w, h = base.size
            gx = unit.gutter_x if unit.gutter_x is not None else w // 2
            pad = max(8, int(0.02 * w))
            if unit.side == "L":
                box = (0, 0, max(1, gx - pad), h)
            else:
                box = (min(w - 1, gx + pad), 0, w, h)
            base = base.crop(box)

        # apply region crop
        w, h = base.size
        if region == "top":
            base = base.crop((0, 0, w, int(0.25 * h)))
        elif region == "bottom":
            base = base.crop((0, int(0.78 * h), w, h))
        elif region == "body":
            base = base.crop((0, int(0.20 * h), w, int(0.82 * h)))
        elif region == "full":
            pass
        else:
            raise ValueError(f"Unknown region: {region}")

        base.save(cache_path)
        return base

    def extract_unit_text(self, unit: UnitRef, *, region: str = "full", y0_ratio: float | None = None) -> str:
        """Extract text from a unit using PDF text layer (if available).

        For scanned PDFs this will usually return an empty string.
        Uses the same unit/region semantics as render_unit.
        """
        if fitz is None:
            return ""
        try:
            page = self._doc.load_page(unit.pdf_page_idx)
            rect = page.rect
            clip = rect

            # apply 2-up side clip
            if unit.layout == "two_up_lr":
                gx = float(unit.gutter_x if unit.gutter_x is not None else rect.width / 2.0)
                pad = max(4.0, 0.02 * rect.width)
                if unit.side == "L":
                    clip = fitz.Rect(rect.x0, rect.y0, rect.x0 + max(1.0, gx - pad), rect.y1)
                else:
                    clip = fitz.Rect(rect.x0 + min(rect.width - 1.0, gx + pad), rect.y0, rect.x1, rect.y1)

            # apply region crop within clip
            h = clip.height
            if region == "top":
                clip = fitz.Rect(clip.x0, clip.y0, clip.x1, clip.y0 + 0.25 * h)
            elif region == "bottom":
                y0 = float(y0_ratio) if y0_ratio is not None else 0.78
                y0 = max(0.0, min(0.98, y0))
                clip = fitz.Rect(clip.x0, clip.y0 + y0 * h, clip.x1, clip.y1)
            elif region == "body":
                clip = fitz.Rect(clip.x0, clip.y0 + 0.20 * h, clip.x1, clip.y0 + 0.82 * h)
            elif region == "full":
                pass
            else:
                raise ValueError(f"Unknown region: {region}")

            txt = page.get_text("text", clip=clip) or ""
            return txt
        except Exception:
            return ""

    def crop_footer_policies(self, img: "Image.Image") -> Dict[str, "Image.Image"]:
        """
        Generate multiple footer crops for page-label reading.

        bottom_outer: bottom-left for L pages, bottom-right for R pages (handled outside).
        Here we just provide generic crops.
        """
        w, h = img.size
        # footer strip
        y0 = int(0.78 * h)
        y1 = h
        strip = img.crop((0, y0, w, y1))
        sw, sh = strip.size
        crops = {
            "bottom_center": strip.crop((int(0.35 * sw), 0, int(0.65 * sw), sh)),
            "bottom_right": strip.crop((int(0.70 * sw), 0, sw, sh)),
            "bottom_left": strip.crop((0, 0, int(0.30 * sw), sh)),
            "bottom_outer": strip,  # caller may pick side-specific outer region
        }
        return crops
