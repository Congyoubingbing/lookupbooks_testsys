# -*- coding: utf-8 -*-
"""Aggregate auto metrics + judge scores for a batch.

Reads:
- _eval/auto_metrics.jsonl
- _eval/judge_scores.jsonl (optional)

Writes:
- _eval/combined.csv
- _eval/summary.json
- _eval/eval_report.md
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def _ensure_eval_dir(batch_dir: Path) -> Path:
    p = Path(batch_dir) / "_eval"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(x)
    except Exception:
        return None


def aggregate_batch(batch_dir: Path) -> Dict[str, Path]:
    batch_dir = Path(batch_dir)
    eval_dir = _ensure_eval_dir(batch_dir)

    auto_rows = _read_jsonl(eval_dir / "auto_metrics.jsonl")
    judge_rows = _read_jsonl(eval_dir / "judge_scores.jsonl")

    # index by qid
    j_map = {str(r.get("qid") or ""): r for r in judge_rows if isinstance(r, dict)}

    combined: List[Dict[str, Any]] = []
    for a in auto_rows:
        qid = str(a.get("qid") or "")
        row = dict(a)
        j = j_map.get(qid)
        if j:
            # prefix judge fields to avoid collisions
            for k, v in j.items():
                if k in {"qid", "mode"}:
                    continue
                out_key = k if str(k).startswith("judge_") else f"judge_{k}"
                row[out_key] = v
        combined.append(row)

    # write combined.csv
    out_csv = eval_dir / "combined.csv"
    fieldnames: List[str] = []
    for r in combined:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    # make important fields first
    pref = [
        "schema_version","batch_id","qid","idx","title","mode","status","spec_ok","retrieval_used","retrieval_rounds","duration_sec",
        "evidence_chars_total","selected_severely_broken",
        "judge_parse_ok","judge_score_overall","judge_score_correctness","judge_score_completeness","judge_score_derivation","judge_score_grounding",
        "judge_score_hallucination_resistance","judge_should_use_tools","judge_confidence",
    ]
    ordered = [k for k in pref if k in fieldnames] + [k for k in fieldnames if k not in pref]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in sorted(combined, key=lambda x: (str(x.get("qid") or ""))):
            flat = {}
            for k in ordered:
                v = r.get(k)
                if isinstance(v, (list, dict)):
                    flat[k] = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    flat[k] = ""
                else:
                    flat[k] = v
            w.writerow(flat)

    # summary.json
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in combined:
        by_mode[str(r.get("mode") or "")].append(r)

    def _rate(rows: List[Dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return sum(1 for r in rows if bool(r.get(key))) / float(len(rows))

    def _mean(rows: List[Dict[str, Any]], key: str) -> float:
        vals = [float(r.get(key)) for r in rows if _safe_float(r.get(key)) is not None]
        return float(mean(vals)) if vals else 0.0

    summary: Dict[str, Any] = {
        "items": len(combined),
        "modes": {},
        "has_judge": bool(judge_rows),
    }

    for m, rows in by_mode.items():
        summary["modes"][m] = {
            "items": len(rows),
            "ok_rate": (sum(1 for r in rows if r.get("status") == "ok") / float(len(rows)) if rows else 0.0),
            "spec_ok_rate": _rate(rows, "spec_ok"),
            "retrieval_rate": _rate(rows, "retrieval_used"),
            "duration_sec_mean": _mean(rows, "duration_sec"),
            "evidence_chars_total_mean": _mean(rows, "evidence_chars_total"),
            "selected_severely_broken_rate": _rate(rows, "selected_severely_broken"),
        }
        # judge means
        if judge_rows:
            for k in [
                "judge_score_overall",
                "judge_score_correctness",
                "judge_score_completeness",
                "judge_score_derivation",
                "judge_score_clarity",
                "judge_score_grounding",
                "judge_score_hallucination_resistance",
                "judge_confidence",
            ]:
                summary["modes"][m][f"{k}_mean"] = _mean(rows, k)
            summary["modes"][m]["judge_should_use_tools_rate"] = _rate(rows, "judge_should_use_tools")

    out_summary = eval_dir / "summary.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # eval_report.md (human readable)
    lines: List[str] = []
    lines.append(f"# Eval Report ({batch_dir.name})")
    lines.append("")
    lines.append(f"- items: {summary['items']}")
    lines.append(f"- has_judge: {summary['has_judge']}")
    lines.append("")

    # table header
    hdr = [
        "mode","items","ok_rate","spec_ok_rate","retrieval_rate","duration_sec_mean","evidence_chars_total_mean","selected_severely_broken_rate"
    ]
    if judge_rows:
        hdr += [
            "judge_score_overall_mean","judge_score_correctness_mean","judge_score_completeness_mean","judge_score_derivation_mean","judge_score_grounding_mean","judge_score_hallucination_resistance_mean","judge_should_use_tools_rate","judge_confidence_mean"
        ]
    lines.append("|" + "|".join(hdr) + "|")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")

    for m in sorted(summary["modes"].keys()):
        d = summary["modes"][m]
        row = [
            m,
            str(d.get("items", 0)),
            f"{d.get('ok_rate', 0.0):.3f}",
            f"{d.get('spec_ok_rate', 0.0):.3f}",
            f"{d.get('retrieval_rate', 0.0):.3f}",
            f"{d.get('duration_sec_mean', 0.0):.3f}",
            f"{d.get('evidence_chars_total_mean', 0.0):.1f}",
            f"{d.get('selected_severely_broken_rate', 0.0):.3f}",
        ]
        if judge_rows:
            row += [
                f"{d.get('judge_score_overall_mean', 0.0):.3f}",
                f"{d.get('judge_score_correctness_mean', 0.0):.3f}",
                f"{d.get('judge_score_completeness_mean', 0.0):.3f}",
                f"{d.get('judge_score_derivation_mean', 0.0):.3f}",
                f"{d.get('judge_score_grounding_mean', 0.0):.3f}",
                f"{d.get('judge_score_hallucination_resistance_mean', 0.0):.3f}",
                f"{d.get('judge_should_use_tools_rate', 0.0):.3f}",
                f"{d.get('judge_confidence_mean', 0.0):.3f}",
            ]
        lines.append("|" + "|".join(row) + "|")

    out_md = eval_dir / "eval_report.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"combined_csv": out_csv, "summary_json": out_summary, "eval_report_md": out_md}

