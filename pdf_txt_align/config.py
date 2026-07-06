from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import yaml

class DotDict(dict):
    """dict with attribute-style access.

    NOTE: __getattr__ MUST raise AttributeError for missing keys.
    This allows Python's built-in getattr(obj, name, default) to work.
    The previous behavior (returning None for missing keys) silently broke
    default handling and caused runtime crashes like float(None).
    """

    def __getattr__(self, k):
        if k in self:
            v = self[k]
            if isinstance(v, dict) and not isinstance(v, DotDict):
                v = DotDict(v)
                # cache nested DotDict to keep identity stable
                self[k] = v
            return v
        raise AttributeError(k)

    def __setattr__(self, k, v):
        # allow cfg.foo = ... in code paths, though config is typically read-only
        self[k] = v

def load_config(path: str | Path) -> DotDict:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    cfg = DotDict(data)
    _apply_config_aliases(cfg)
    return cfg


def _apply_config_aliases(cfg: DotDict) -> None:
    """Normalize missing config keys (aliases) to reduce silent misconfigurations.

    Rationale:
    - Some modules evolved key names (e.g., TOC scanning knobs).
    - Users often keep older config files; code should remain compatible.

    This function only *fills in missing keys*; it will not override explicit
    values provided by the user.
    """

    def _sec(name: str) -> DotDict:
        try:
            return getattr(cfg, name)
        except Exception:
            return DotDict({})

    def _get(sec: str, key: str):
        try:
            return getattr(_sec(sec), key)
        except Exception:
            return None

    def _set_if_missing(sec: str, key: str, value) -> None:
        if value is None:
            return
        s = _sec(sec)
        try:
            getattr(s, key)
            return
        except Exception:
            pass
        try:
            setattr(s, key, value)
        except Exception:
            try:
                s[key] = value
            except Exception:
                pass

    # --- PDF / TOC ---
    _set_if_missing("pdf", "toc_scan_score_threshold", _get("pdf", "toc_score_threshold"))
    _set_if_missing("pdf", "toc_scan_stride_pass1", _get("pdf", "toc_stride_pass1"))
    _set_if_missing("pdf", "toc_scan_stride_pass2", _get("pdf", "toc_stride_pass2"))
    _set_if_missing("pdf", "toc_range_backward_units", _get("pdf", "toc_expand_backward_units"))
    _set_if_missing("pdf", "toc_range_forward_units", _get("pdf", "toc_expand_forward_units"))

    # Notes/docs synonym: page1_scan_early_units
    _set_if_missing("pdf", "page1_scan_early_units", _get("pdf", "page1_scan_first_units"))

    # Page-1 inference guards
    _set_if_missing("pdf", "infer_page1_high_pages_threshold", 30)
    _set_if_missing("pdf", "infer_page1_untrusted_conf_cap", 0.7)

    # --- Align / boundaries ---
    # These two were historically hard-coded defaults in pipeline.py; expose as config.
    if _get("align", "snippet_min_score") is None:
        _set_if_missing("align", "snippet_min_score", 0.20)
    if _get("align", "snippet_soft_min_score") is None:
        _set_if_missing("align", "snippet_soft_min_score", 0.15)

    # Safe-cut / boundary snap parameters
    _set_if_missing("align", "safe_cut_midline_penalty", 250)
    _set_if_missing("align", "safe_cut_strong_chapter_bonus", 800)

    # Lead-offset guard (prevent high local snippet score from matching far away from boundary)
    _set_if_missing("align", "lead_max_offset", 600)
    _set_if_missing("align", "lead_scan_chars", 20000)
    _set_if_missing("align", "lead_offset_strategy", "shift")

    # No-exact-occurrence fallback: approximate opening alignment within candidate chunks
    _set_if_missing("align", "opening_align_min_score", 55.0)

    # Prefer TeX structural heading boundaries when they are at least mildly credible.
    # This addresses the recurring failure mode where a high-scoring anchor/snippet match
    # lands deep into the chapter (or at the next chapter heading), causing cross-chapter cuts.
    _set_if_missing("align", "tex_heading_pref_min_conf", 0.05)

    # TeX headings like "CHAPTER 13" / "第九章" contain little semantic text, so title
    # similarity can be deceptively low despite being the correct structural boundary.
    _set_if_missing("align", "tex_numbered_heading_conf_floor", 0.78)
    # v12.4.7 compatibility / quality-first defaults
    _set_if_missing("align", "tex_parse_allow_starred_headings", True)
    _set_if_missing("align", "tex_chapter_detect_enable_allcaps_section_star", True)
    _set_if_missing("align", "tex_chapter_detect_allow_cn_chapter_spaces", True)
    _set_if_missing("align", "tex_section_like_integer_dot_demote", True)
    _set_if_missing("align", "tex_toc_title_promotion_enable", True)
    _set_if_missing("align", "tex_toc_title_promotion_min_sim", 0.72)
    _set_if_missing("align", "tex_heading_lock_dual_boundary_enable", True)
    _set_if_missing("align", "tex_heading_lock_reclaim_next_heading", True)
    _set_if_missing("align", "safe_cut_avoid_midword", True)
    _set_if_missing("align", "safe_cut_midword_penalty", 0.35)
    _set_if_missing("align", "backmatter_trim_enable", True)
    _set_if_missing("align", "backmatter_trim_keywords", [
        "\\section*{REFERENCES}",
        "\\section*{References}",
        "\\section*{SUBJECT INDEX}",
        "\\section*{AUTHOR INDEX}",
        "\\section*{参考文献}",
        "\\section*{索引}",
    ])
    _set_if_missing("align", "pdf_text_probe_min_sim", 0.75)
    _set_if_missing("pdf", "unit_image_dpi_low", _get("pdf", "dpi_low") if _get("pdf", "dpi_low") is not None else 160)
    _set_if_missing("pdf", "unit_image_dpi_high", _get("pdf", "dpi_high") if _get("pdf", "dpi_high") is not None else 320)
    _set_if_missing("llm", "vl_boundary_verify_max_tokens", 300)
    _set_if_missing("llm", "vl_boundary_relocate_max_tokens", 400)

    # Backward-anchor guard: optional forward snap to structural heading
    _set_if_missing("align", "backward_anchor_heading_snap_limit", 40000)
