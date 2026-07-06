# -*- coding: utf-8 -*-
"""Compare multiple batch eval outputs.

This utility is designed for running M1..M8 as separate batch dirs.
It loads each batch_dir/_eval/combined.csv and produces a comparison folder:

runs/_compare/<compare_id>/
  combined_long.csv
  per_question_wide.csv
  delta_vs_baseline.csv

Note: This is a lightweight CSV-only implementation (no pandas).
"""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_csv(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    with p.open('r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        return [dict(x) for x in r]


def _write_csv(p: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})


def _safe_float(x) -> Optional[float]:
    try:
        if x is None or x == '':
            return None
        return float(x)
    except Exception:
        return None


def compare_batches(batch_dirs: List[Path], *, out_dir: Path, baseline_mode: str = 'M2') -> Dict[str, Path]:
    rows_long: List[Dict[str, Any]] = []
    for bd in batch_dirs:
        bd = Path(bd)
        comb = bd / '_eval' / 'combined.csv'
        rows = _read_csv(comb)
        for r in rows:
            r2 = dict(r)
            r2['batch_id'] = bd.name
            r2['batch_dir'] = str(bd.resolve())
            rows_long.append(r2)

    # write long
    fieldnames_long: List[str] = []
    for r in rows_long:
        for k in r.keys():
            if k not in fieldnames_long:
                fieldnames_long.append(k)
    out_long = out_dir / 'combined_long.csv'
    _write_csv(out_long, rows_long, fieldnames_long)

    # wide per qid+mode
    # key: qid, value: dict
    wide: Dict[str, Dict[str, Any]] = {}
    for r in rows_long:
        qid = str(r.get('qid') or '')
        mode = str(r.get('mode') or '')
        if not qid or not mode:
            continue
        # Disambiguate M8 k-sweep runs: otherwise M8_k1/k3/k5 will be folded into a single 'M8' column.
        mode_key = mode
        try:
            if mode.strip().upper() == 'M8':
                k = r.get('topk_sections') or r.get('section_budget_used') or r.get('topk_pairs') or r.get('topk_books') or ''
                k_int = int(float(k)) if str(k).strip() != '' else 0
                if k_int > 0:
                    mode_key = f"M8_k{k_int}"
        except Exception:
            mode_key = mode
        w = wide.setdefault(qid, {'qid': qid})
        # copy key metrics
        for k in ['status','spec_ok','retrieval_used','retrieval_rounds','duration_sec','evidence_chars_total','judge_score_overall','judge_score_correctness','judge_score_completeness','judge_score_derivation','judge_score_grounding','judge_score_hallucination_resistance']:
            if k in r:
                w[f'{mode_key}.{k}'] = r.get(k)

    wide_rows = [wide[q] for q in sorted(wide.keys())]
    fieldnames_wide: List[str] = ['qid']
    # collect columns
    cols = set()
    for r in wide_rows:
        for k in r.keys():
            if k != 'qid':
                cols.add(k)
    fieldnames_wide += sorted(cols)
    out_wide = out_dir / 'per_question_wide.csv'
    _write_csv(out_wide, wide_rows, fieldnames_wide)

    # delta vs baseline
    base = baseline_mode.strip().upper()
    deltas: List[Dict[str, Any]] = []
    for r in wide_rows:
        qid = r.get('qid')
        base_overall = _safe_float(r.get(f'{base}.judge_score_overall'))
        if base_overall is None:
            continue
        for k, v in r.items():
            if not k.endswith('.judge_score_overall') or k.startswith(f'{base}.'):
                continue
            mode = k.split('.')[0]
            ov = _safe_float(v)
            if ov is None:
                continue
            deltas.append({
                'qid': qid,
                'mode': mode,
                'baseline_mode': base,
                'delta_judge_score_overall': round(ov - base_overall, 6),
                'baseline_overall': base_overall,
                'mode_overall': ov,
            })

    out_delta = out_dir / 'delta_vs_baseline.csv'
    _write_csv(out_delta, deltas, ['qid','mode','baseline_mode','delta_judge_score_overall','baseline_overall','mode_overall'])

    return {'combined_long_csv': out_long, 'per_question_wide_csv': out_wide, 'delta_vs_baseline_csv': out_delta}

