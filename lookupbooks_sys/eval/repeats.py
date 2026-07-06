# -*- coding: utf-8 -*-
"""Aggregate repeated runs across rounds/batches.

Produces:
  - repeat_long.csv         (one row per qid x mode x batch)
  - repeat_agg.csv          (one row per qid x mode with aggregate stats)
  - repeat_mode_summary.csv (one row per mode aggregated across questions)
  - repeat_summary.json
"""
from __future__ import annotations

import csv
import datetime
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

def _read_csv(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    with p.open('r', encoding='utf-8') as f:
        return [dict(x) for x in csv.DictReader(f)]

def _write_csv(p: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == '':
            return None
        return float(x)
    except Exception:
        return None

def _safe_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    s = str(x or '').strip().lower()
    if s in {'true', '1', 'yes'}:
        return True
    if s in {'false', '0', 'no'}:
        return False
    return None

def _stats(vals: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not vals:
        return None, None, None, None
    if len(vals) == 1:
        v = float(vals[0])
        return v, 0.0, v, v
    return float(mean(vals)), float(pstdev(vals)), float(min(vals)), float(max(vals))

def aggregate_repeat_batches(batch_dirs: List[Path], *, out_dir: Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_long: List[Dict[str, Any]] = []
    for bd in [Path(x) for x in batch_dirs]:
        comb = bd / '_eval' / 'combined.csv'
        if not comb.exists():
            continue
        rows = _read_csv(comb)
        for r in rows:
            r2 = dict(r)
            r2['batch_id'] = bd.name
            r2['batch_dir'] = str(bd.resolve())
            rows_long.append(r2)

    # long rows
    fieldnames_long: List[str] = []
    for r in rows_long:
        for k in r.keys():
            if k not in fieldnames_long:
                fieldnames_long.append(k)
    out_long = out_dir / 'repeat_long.csv'
    _write_csv(out_long, rows_long, fieldnames_long)

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows_long:
        qid = str(r.get('qid') or '')
        mode = str(r.get('mode') or '')
        if qid and mode:
            groups[(qid, mode)].append(r)

    metrics = [
        'judge_score_overall', 'judge_score_correctness', 'judge_score_completeness',
        'judge_score_derivation', 'judge_score_clarity', 'judge_score_grounding',
        'judge_score_hallucination_resistance', 'duration_sec', 'evidence_chars_total',
        'retrieval_rounds',
    ]
    bool_metrics = ['spec_ok', 'retrieval_used', 'selected_severely_broken']

    agg_rows: List[Dict[str, Any]] = []
    for (qid, mode), rows in sorted(groups.items()):
        agg: Dict[str, Any] = {'qid': qid, 'mode': mode, 'n_runs': len(rows)}
        ok_runs = sum(1 for r in rows if str(r.get('status') or '') == 'ok')
        agg['ok_runs'] = ok_runs
        agg['failed_runs'] = len(rows) - ok_runs
        for bm in bool_metrics:
            vals = [_safe_bool(r.get(bm)) for r in rows]
            vals = [v for v in vals if v is not None]
            agg[f'{bm}_rate'] = round(sum(1 for v in vals if v) / len(vals), 6) if vals else ''
        for m in metrics:
            vals = [_safe_float(r.get(m)) for r in rows]
            vals = [v for v in vals if v is not None]
            mean_v, std_v, min_v, max_v = _stats(vals)
            agg[f'{m}_mean'] = '' if mean_v is None else round(mean_v, 6)
            agg[f'{m}_std'] = '' if std_v is None else round(std_v, 6)
            agg[f'{m}_min'] = '' if min_v is None else round(min_v, 6)
            agg[f'{m}_max'] = '' if max_v is None else round(max_v, 6)
        agg_rows.append(agg)

    fieldnames_agg: List[str] = []
    for r in agg_rows:
        for k in r.keys():
            if k not in fieldnames_agg:
                fieldnames_agg.append(k)
    out_agg = out_dir / 'repeat_agg.csv'
    _write_csv(out_agg, agg_rows, fieldnames_agg)

    # mode summary over agg rows
    mode_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in agg_rows:
        mode_groups[str(r.get('mode') or '')].append(r)
    mode_rows: List[Dict[str, Any]] = []
    for mode, rows in sorted(mode_groups.items()):
        d: Dict[str, Any] = {'mode': mode, 'questions': len(rows)}
        for key in [
            'judge_score_overall_mean', 'judge_score_correctness_mean', 'judge_score_completeness_mean',
            'judge_score_derivation_mean', 'judge_score_clarity_mean', 'judge_score_grounding_mean',
            'judge_score_hallucination_resistance_mean', 'duration_sec_mean', 'evidence_chars_total_mean',
            'spec_ok_rate', 'retrieval_used_rate', 'selected_severely_broken_rate'
        ]:
            vals = [_safe_float(r.get(key)) for r in rows]
            vals = [v for v in vals if v is not None]
            d[key] = '' if not vals else round(float(mean(vals)), 6)
        mode_rows.append(d)
    fieldnames_mode: List[str] = []
    for r in mode_rows:
        for k in r.keys():
            if k not in fieldnames_mode:
                fieldnames_mode.append(k)
    out_mode = out_dir / 'repeat_mode_summary.csv'
    _write_csv(out_mode, mode_rows, fieldnames_mode)

    summary = {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'batches': [str(Path(x).resolve()) for x in batch_dirs],
        'items_long': len(rows_long),
        'items_agg': len(agg_rows),
        'modes': sorted(mode_groups.keys()),
    }
    out_json = out_dir / 'repeat_summary.json'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return {
        'repeat_long_csv': out_long,
        'repeat_agg_csv': out_agg,
        'repeat_mode_summary_csv': out_mode,
        'repeat_summary_json': out_json,
    }
