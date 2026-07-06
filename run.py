#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import traceback
import time
import sys
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pdf_txt_align.api_pool import ApiPool
from pdf_txt_align.config import load_config
from pdf_txt_align.pipeline import process_one_book
from pdf_txt_align.utils import dump_json, ensure_dir, env_keys, now_ts, sha1_bytes

from lookupbooks_sys.library import (
    BookLibrary,
    BookIndexEntry,
    import_from_pdf_txt_align_outputs,
    add_book_from_output_dir,
    remove_book_from_library,
)
from lookupbooks_sys.library_registry import sync_registry, read_registry
from lookupbooks_sys.summarizer import build_book_summaries, build_chapter_summaries_for_book
from lookupbooks_sys.qa_agent import MultiBookQAAgent
from lookupbooks_sys.reporting import (
    ReportItem,
    build_item_from_qa_result,
    render_markdown_index,
    render_markdown_report,
    write_markdown_report,
)


# -------------------------
# Helpers
# -------------------------
def find_pairs(pairs_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    pdfs = sorted(pairs_dir.glob("*.pdf"))
    for pdf in pdfs:
        stem = pdf.stem
        tex = pairs_dir / f"{stem}.tex"
        txt = pairs_dir / f"{stem}.txt"
        if tex.exists():
            pairs.append((pdf, tex))
        elif txt.exists():
            pairs.append((pdf, txt))
    return pairs


def _list_book_output_dirs(outputs_root: Path) -> List[Path]:
    if not outputs_root.exists() or not outputs_root.is_dir():
        return []
    out: List[Path] = []
    for p in outputs_root.iterdir():
        if p.is_dir() and ((p / "chapters").exists() or (p / "sections").exists() or (p / "book_overview.json").exists() or (p / "chapter_overview.json").exists()):
            out.append(p)
    return sorted(out, key=lambda x: x.name.lower())


def _suggest_similar_dirs(outputs_root: Path, needle: str, max_items: int = 10) -> List[str]:
    """Return a short list of candidate folder names under outputs_root."""
    needle = (needle or "").strip().lower()
    dirs = _list_book_output_dirs(outputs_root)
    names = [d.name for d in dirs]
    if not needle:
        return names[:max_items]

    # simple substring ranking
    scored: List[Tuple[int, str]] = []
    for n in names:
        nl = n.lower()
        score = 0
        if needle in nl:
            score += 100
        # overlap count
        score += sum(1 for ch in set(needle) if ch in nl)
        scored.append((score, n))
    scored.sort(key=lambda x: (-x[0], x[1].lower()))
    return [n for s, n in scored[:max_items] if s > 0] or names[:max_items]


def _load_batch_question_meta(batch_dir: Path) -> Dict[str, Dict[str, Any]]:
    parsed = batch_dir / "parsed_questions.json"
    if not parsed.exists():
        return {}
    try:
        data = json.loads(parsed.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("qid") or "").strip()
            if qid:
                out[qid] = item
    return out


def _load_batch_question_list(batch_dir: Path) -> List[Dict[str, Any]]:
    parsed = batch_dir / "parsed_questions.json"
    if not parsed.exists():
        return []
    try:
        data = json.loads(parsed.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and str(item.get("qid") or "").strip():
                out.append(item)
    return out


def _merge_question_lists(existing: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for src in (existing or []):
        if not isinstance(src, dict):
            continue
        qid = str(src.get("qid") or "").strip()
        if not qid:
            continue
        if qid not in merged:
            order.append(qid)
        merged[qid] = src
    for src in (new_items or []):
        if not isinstance(src, dict):
            continue
        qid = str(src.get("qid") or "").strip()
        if not qid:
            continue
        if qid not in merged:
            order.append(qid)
        merged[qid] = src
    def _idx(item: Dict[str, Any]) -> int:
        try:
            return int(item.get("idx") or 0)
        except Exception:
            return 0
    ordered = [merged[qid] for qid in order]
    ordered.sort(key=_idx)
    return ordered


def _load_batch_summary(batch_dir: Path) -> Dict[str, Any]:
    p = batch_dir / "batch_summary.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _summary_results_by_qid(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in summary.get("results") or []:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("qid") or "").strip()
        if qid:
            out[qid] = item
    return out


def _ordered_results_from_qmeta(qmeta_list: List[Dict[str, Any]], results_by_qid: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in qmeta_list or []:
        qid = str(item.get("qid") or "").strip()
        if qid and qid in results_by_qid and qid not in seen:
            ordered.append(results_by_qid[qid])
            seen.add(qid)
    for qid, row in results_by_qid.items():
        if qid not in seen:
            ordered.append(row)
    return ordered


def _infer_batch_mode_and_overrides(batch_dir: Path) -> Tuple[str, Optional[int]]:
    summary = _load_batch_summary(batch_dir)
    modes = {str(r.get("mode") or "").strip() for r in (summary.get("results") or []) if isinstance(r, dict) and str(r.get("mode") or "").strip()}
    mode = next(iter(modes)) if len(modes) == 1 else ""
    topk_sections: Optional[int] = None

    if not mode:
        for q_dir in sorted([p for p in batch_dir.iterdir() if p.is_dir() and p.name.upper().startswith("Q")]):
            qa_path = q_dir / "qa_result.json"
            if qa_path.exists():
                try:
                    payload = json.loads(qa_path.read_text(encoding="utf-8"))
                    mode = str(payload.get("test_mode") or "").strip()
                    if mode == "M8":
                        try:
                            topk_sections = int(payload.get("topk_sections")) if payload.get("topk_sections") is not None else None
                        except Exception:
                            topk_sections = None
                    break
                except Exception:
                    pass

    name = batch_dir.name
    if not mode:
        m = re.search(r'(?:^|_)(M[1-9])(?:$|_)', name)
        if m:
            mode = m.group(1)
        if '_M8_k' in name:
            mode = 'M8'

    if topk_sections is None:
        m = re.search(r'_M8_k(\d+)$', name)
        if m:
            topk_sections = int(m.group(1))

    return mode, topk_sections


def _inspect_batch_failures(batch_dir: Path, *, expect_judge: bool = True) -> Dict[str, Any]:
    batch_dir = Path(batch_dir)
    qmeta_list = _load_batch_question_list(batch_dir)
    qmeta_map = {str(x.get("qid") or "").strip(): x for x in qmeta_list if isinstance(x, dict)}
    summary = _load_batch_summary(batch_dir)
    summary_by_qid = _summary_results_by_qid(summary)

    answer_failed: List[str] = []
    answer_failed_reasons: Dict[str, List[str]] = {}
    judge_failed: List[str] = []
    judge_failed_reasons: Dict[str, List[str]] = {}

    qids = sorted(set(list(qmeta_map.keys()) + list(summary_by_qid.keys())))
    eval_dir = batch_dir / '_eval'
    outputs_dir = eval_dir / 'judge_outputs'

    for qid in qids:
        q_dir = batch_dir / qid
        qa_path = q_dir / 'qa_result.json'
        reasons: List[str] = []
        if not q_dir.exists():
            reasons.append('q_dir_missing')
        if not qa_path.exists():
            reasons.append('qa_result_missing')
        else:
            try:
                json.loads(qa_path.read_text(encoding='utf-8'))
            except Exception:
                reasons.append('qa_result_invalid_json')
        row = summary_by_qid.get(qid)
        if isinstance(row, dict) and str(row.get('status') or '').strip().lower() == 'failed':
            reasons.append('batch_summary_failed')
        if reasons:
            answer_failed.append(qid)
            answer_failed_reasons[qid] = sorted(set(reasons))
            continue

        if expect_judge:
            jreasons: List[str] = []
            out_path = outputs_dir / f'{qid}.json'
            if not out_path.exists():
                jreasons.append('judge_output_missing')
            else:
                try:
                    payload = json.loads(out_path.read_text(encoding='utf-8'))
                    parsed = payload.get('parsed') if isinstance(payload, dict) else None
                    valerrs = payload.get('validation_errors') if isinstance(payload, dict) else None
                    if not isinstance(parsed, dict):
                        jreasons.append('judge_output_invalid')
                    else:
                        key_issues = parsed.get('key_issues') if isinstance(parsed.get('key_issues'), list) else []
                        unsupported = parsed.get('unsupported_claims') if isinstance(parsed.get('unsupported_claims'), list) else []
                        signals = set([str(x) for x in (valerrs or [])] + [str(x) for x in key_issues] + [str(x) for x in unsupported])
                        if 'judge_request_failed' in signals or 'judge_parse_failed' in signals:
                            jreasons.append('judge_failed_sentinel')
                except Exception:
                    jreasons.append('judge_output_invalid_json')
            if jreasons:
                judge_failed.append(qid)
                judge_failed_reasons[qid] = sorted(set(jreasons))

    return {
        'batch_dir': str(batch_dir.resolve()),
        'batch_id': batch_dir.name,
        'question_count': len(qids),
        'answer_failed_qids': sorted(set(answer_failed)),
        'answer_failed_reasons': answer_failed_reasons,
        'judge_failed_qids': sorted(set(judge_failed)),
        'judge_failed_reasons': judge_failed_reasons,
    }


def _collect_batch_dirs(batch_dirs: List[str], glob_pat: str) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for raw in batch_dirs or []:
        p = Path(raw)
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    if glob_pat:
        for raw in sorted(Path().glob(glob_pat)):
            if raw.is_dir():
                key = str(raw.resolve())
                if key not in seen:
                    out.append(raw)
                    seen.add(key)
    return out


def _build_q0_markdown(items: List[Dict[str, Any]], *, tag: str = 'q0') -> str:
    blocks: List[str] = []
    for item in items:
        qid = str(item.get('qid') or '').strip()
        title = str(item.get('title') or '').strip()
        question = str(item.get('question') or '').strip()
        if not qid or not question:
            continue
        header = f"## {qid}"
        if title:
            header += f"｜{title}"
        blocks.append(header)
        blocks.append(f"```{tag}\n{question}\n```")
        blocks.append('')
    return '\n'.join(blocks).strip() + '\n'

def _recover_failed_question_items(batch_dir: Path, batch_file: str, qids: List[str], qmeta_map: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Best-effort recovery of failed question metadata for repair reruns.

    Order of fallback:
    1) parsed_questions.json entry already in qmeta_map
    2) per-question q0.txt under batch_dir/Qxxx
    3) reparsing the original batch_file markdown
    """
    resolved: Dict[str, Dict[str, Any]] = {}
    unresolved: List[str] = []

    parsed_from_batch: Dict[str, Dict[str, Any]] = {}
    if batch_file:
        try:
            bf = Path(batch_file)
            if bf.exists():
                md_text = bf.read_text(encoding='utf-8')
                parsed_from_batch = {str(x.get('qid') or '').strip(): x for x in _parse_q0_blocks(md_text, tag='q0') if isinstance(x, dict)}
        except Exception:
            parsed_from_batch = {}

    for pos, qid in enumerate(qids, start=1):
        qid = str(qid or '').strip()
        if not qid:
            continue
        item = qmeta_map.get(qid)
        if isinstance(item, dict):
            qtext = str(item.get('question') or '').strip()
            if qtext:
                resolved[qid] = dict(item)
                continue

        # fallback 1: q0.txt in existing q_dir
        q_dir = batch_dir / qid
        q0 = q_dir / 'q0.txt'
        if q0.exists():
            try:
                qtext = q0.read_text(encoding='utf-8').strip()
            except Exception:
                qtext = ''
            if qtext:
                base = dict(qmeta_map.get(qid) or {})
                base.setdefault('idx', pos)
                base['qid'] = qid
                base['question'] = qtext
                base.setdefault('title', str(base.get('title') or ''))
                resolved[qid] = base
                continue

        # fallback 2: reparsed original batch markdown
        item2 = parsed_from_batch.get(qid)
        if isinstance(item2, dict) and str(item2.get('question') or '').strip():
            resolved[qid] = dict(item2)
            continue

        unresolved.append(qid)

    ordered = [resolved[qid] for qid in qids if qid in resolved]
    return ordered, unresolved


def _namespace_for_batch_rerun(*, batch_file: str, batch_id: str, runs_dir: str, agent_config: str, library_root: str, outputs_root: str, workers: int, max_attempts: int, mode: str, topk_sections: Optional[int], verbose: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        suite='', suite_id='repair', suite_runs_dir='./runs/suite', suite_m8_ks='1,3,5', suite_judge=False, suite_rubric='', suite_judge_workers=0, suite_skip_existing_judge=False, suite_stop_on_error=False, suite_baseline_mode='M2',
        batch_file=batch_file, tag='q0', agent_config=agent_config, outputs_root=outputs_root, library_root=library_root, runs_dir=runs_dir, batch_id=batch_id, workers=workers, max_attempts=max_attempts, start=1, end=0, stop_on_error=False, auto_summarize=False, verbose=verbose, test_mode=mode, topk_books=None, topk_pairs=None, topk_sections=topk_sections
    )


def _namespace_for_eval_repair(*, batch_dir: str, agent_config: str, judge_workers: int) -> argparse.Namespace:
    return argparse.Namespace(
        batch_dir=batch_dir, agent_config=agent_config, judge=True, rubric='', judge_workers=judge_workers, skip_existing=True, library_root='', test_mode=''
    )


def cmd_scan_batch_failures(args) -> None:
    batch_dirs = _collect_batch_dirs(list(getattr(args, 'batch_dir', []) or []), str(getattr(args, 'glob', '') or ''))
    if not batch_dirs:
        raise SystemExit('scan-batch-failures requires at least 1 --batch-dir or --glob')
    out_dir = ensure_dir(Path(args.out_dir))
    items: List[Dict[str, Any]] = []
    for bd in batch_dirs:
        rec = _inspect_batch_failures(bd, expect_judge=not bool(getattr(args, 'no_expect_judge', False)))
        items.append(rec)
    js = out_dir / 'failed_items.json'
    js.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
    csv_path = out_dir / 'failed_items.csv'
    rows: List[Dict[str, Any]] = []
    for rec in items:
        for qid in rec.get('answer_failed_qids') or []:
            rows.append({'batch_id': rec.get('batch_id'), 'batch_dir': rec.get('batch_dir'), 'qid': qid, 'kind': 'answer', 'reasons': '|'.join(rec.get('answer_failed_reasons', {}).get(qid, []))})
        for qid in rec.get('judge_failed_qids') or []:
            rows.append({'batch_id': rec.get('batch_id'), 'batch_dir': rec.get('batch_dir'), 'qid': qid, 'kind': 'judge', 'reasons': '|'.join(rec.get('judge_failed_reasons', {}).get(qid, []))})
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['batch_id', 'batch_dir', 'qid', 'kind', 'reasons'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {
        'batches': len(items),
        'answer_failed_total': sum(len(rec.get('answer_failed_qids') or []) for rec in items),
        'judge_failed_total': sum(len(rec.get('judge_failed_qids') or []) for rec in items),
        'json': str(js.resolve()),
        'csv': str(csv_path.resolve()),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[FAIL-CHECK] outputs:', summary)


def cmd_repair_batch_failures(args) -> None:
    batch_dirs = _collect_batch_dirs(list(getattr(args, 'batch_dir', []) or []), str(getattr(args, 'glob', '') or ''))
    if not batch_dirs:
        raise SystemExit('repair-batch-failures requires at least 1 --batch-dir or --glob')
    out_dir = ensure_dir(Path(args.out_dir))
    repair_records: List[Dict[str, Any]] = []
    for bd in batch_dirs:
        bd = Path(bd)
        before = _inspect_batch_failures(bd, expect_judge=True)
        mode, topk_sections = _infer_batch_mode_and_overrides(bd)
        summary = _load_batch_summary(bd)
        batch_file = str(summary.get('batch_file') or '')
        qmeta_list = _load_batch_question_list(bd)
        qmeta_map = {str(x.get('qid') or '').strip(): x for x in qmeta_list if isinstance(x, dict)}

        rec: Dict[str, Any] = {
            'batch_id': bd.name,
            'batch_dir': str(bd.resolve()),
            'mode': mode,
            'topk_sections': topk_sections,
            'before': before,
            'answer_rerun_qids': [],
            'judge_rerun_qids': [],
        }

        answer_qids = list(before.get('answer_failed_qids') or [])
        rec['answer_unresolved_qids'] = []
        if answer_qids:
            if not batch_file:
                rec['answer_unresolved_qids'] = list(answer_qids)
                print(f'[REPAIR-WARN] batch_summary.json missing batch_file for {bd}; cannot rerun answer qids={answer_qids}', flush=True)
            else:
                missing_items, unresolved = _recover_failed_question_items(bd, batch_file, answer_qids, qmeta_map)
                rec['answer_unresolved_qids'] = list(unresolved)
                rerun_qids = [str(x.get('qid') or '').strip() for x in missing_items if isinstance(x, dict)]
                if unresolved:
                    print(f'[REPAIR-WARN] recovered only part of failed qids for {bd}; unresolved={unresolved}', flush=True)
                if missing_items:
                    for qid in rerun_qids:
                        q_dir = bd / qid
                        if q_dir.exists():
                            shutil.rmtree(q_dir, ignore_errors=True)
                        jout = bd / '_eval' / 'judge_outputs' / f'{qid}.json'
                        jin = bd / '_eval' / 'judge_inputs' / f'{qid}.json'
                        for p in [jout, jin]:
                            if p.exists():
                                try:
                                    p.unlink()
                                except Exception:
                                    pass
                    temp_md = out_dir / f'{bd.name}__failed_q0.md'
                    temp_md.write_text(_build_q0_markdown(missing_items, tag='q0'), encoding='utf-8')
                    ns = _namespace_for_batch_rerun(
                        batch_file=str(temp_md),
                        batch_id=bd.name,
                        runs_dir=str(bd.parent),
                        agent_config=str(args.agent_config),
                        library_root=str(args.library_root),
                        outputs_root=str(getattr(args, 'outputs_root', './outputs') or './outputs'),
                        workers=int(getattr(args, 'workers', 24) or 24),
                        max_attempts=int(getattr(args, 'max_attempts', 3) or 3),
                        mode=mode,
                        topk_sections=topk_sections,
                        verbose=bool(getattr(args, 'verbose', False)),
                    )
                    print(f'[REPAIR] rerun answers: batch={bd.name} qids={rerun_qids}', flush=True)
                    cmd_batch_ask(ns)
                    rec['answer_rerun_qids'] = rerun_qids
                else:
                    print(f'[REPAIR-WARN] no recoverable failed answer qids for {bd}: {answer_qids}', flush=True)

        after_answer = _inspect_batch_failures(bd, expect_judge=True)
        judge_qids = sorted(set(list(after_answer.get('judge_failed_qids') or []) + list(rec['answer_rerun_qids'] or [])))
        if judge_qids:
            for qid in judge_qids:
                for p in [bd / '_eval' / 'judge_outputs' / f'{qid}.json', bd / '_eval' / 'judge_inputs' / f'{qid}.json']:
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass
            print(f'[REPAIR] rerun judge: batch={bd.name} qids={judge_qids}', flush=True)
            ns_eval = _namespace_for_eval_repair(batch_dir=str(bd), agent_config=str(args.agent_config), judge_workers=int(getattr(args, 'judge_workers', 24) or 24))
            cmd_eval_batch(ns_eval)
            rec['judge_rerun_qids'] = judge_qids

        after = _inspect_batch_failures(bd, expect_judge=True)
        rec['after'] = after
        repair_records.append(rec)

    js = out_dir / 'repair_results.json'
    js.write_text(json.dumps(repair_records, ensure_ascii=False, indent=2), encoding='utf-8')
    csv_path = out_dir / 'repair_results.csv'
    rows: List[Dict[str, Any]] = []
    for rec in repair_records:
        rows.append({
            'batch_id': rec.get('batch_id'),
            'answer_failed_before': len(rec.get('before', {}).get('answer_failed_qids') or []),
            'judge_failed_before': len(rec.get('before', {}).get('judge_failed_qids') or []),
            'answer_rerun': len(rec.get('answer_rerun_qids') or []),
            'judge_rerun': len(rec.get('judge_rerun_qids') or []),
            'answer_failed_after': len(rec.get('after', {}).get('answer_failed_qids') or []),
            'judge_failed_after': len(rec.get('after', {}).get('judge_failed_qids') or []),
        })
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['batch_id', 'answer_failed_before', 'judge_failed_before', 'answer_rerun', 'judge_rerun', 'answer_failed_after', 'judge_failed_after'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {
        'batches': len(repair_records),
        'answer_failed_before_total': sum(len(rec.get('before', {}).get('answer_failed_qids') or []) for rec in repair_records),
        'judge_failed_before_total': sum(len(rec.get('before', {}).get('judge_failed_qids') or []) for rec in repair_records),
        'answer_failed_after_total': sum(len(rec.get('after', {}).get('answer_failed_qids') or []) for rec in repair_records),
        'judge_failed_after_total': sum(len(rec.get('after', {}).get('judge_failed_qids') or []) for rec in repair_records),
        'json': str(js.resolve()),
        'csv': str(csv_path.resolve()),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[REPAIR] outputs:', summary)


def _refresh_batch_markdown_reports(batch_dir: Path, *, batch_title: str, batch_title_audit: str, meta: Dict[str, Any]) -> None:
    q_meta = _load_batch_question_meta(batch_dir)
    report_items: List[ReportItem] = []
    q_dirs = sorted([p for p in batch_dir.iterdir() if p.is_dir() and p.name.upper().startswith("Q")])
    for q_dir in q_dirs:
        qid = q_dir.name
        info = q_meta.get(qid, {})
        title = str(info.get("title") or "")
        qtext = str(info.get("question") or "")
        q0 = q_dir / "q0.txt"
        if not qtext and q0.exists():
            try:
                qtext = q0.read_text(encoding="utf-8")
            except Exception:
                qtext = ""
        qa_path = q_dir / "qa_result.json"
        if qa_path.exists():
            payload = json.loads(qa_path.read_text(encoding="utf-8"))
            item = build_item_from_qa_result(qid=qid, title=title, question=qtext, qa_result_payload=payload, q_dir=q_dir)
            report_items.append(item)
            single_meta = dict(meta)
            single_meta.setdefault("test_mode", str(payload.get("test_mode") or meta.get("test_mode") or ""))
            write_markdown_report(q_dir / "report.md", render_markdown_report([item], title=f"lookupbooks_sys 批量问答单题报告（{qid}）", meta=single_meta))
            write_markdown_report(q_dir / "report_audit.md", render_markdown_report([item], title=f"lookupbooks_sys 批量问答单题报告（思考过程审查版｜{qid}）", meta=single_meta, mode="audit"))
        else:
            err = "qa_result.json missing"
            errp = q_dir / "error.txt"
            if errp.exists():
                err = errp.read_text(encoding="utf-8", errors="ignore")[:1000]
            report_items.append(ReportItem(qid=qid, title=title, question=qtext, status="failed", error=err))

    write_markdown_report(batch_dir / "batch_report.md", render_markdown_report(report_items, title=batch_title, meta=meta))
    write_markdown_report(batch_dir / "batch_report_audit.md", render_markdown_report(report_items, title=batch_title_audit, meta=meta, mode="audit"))
    idx_md = render_markdown_index(report_items, title="lookupbooks_sys 批量问答索引（推荐先看）", meta=meta)
    write_markdown_report(batch_dir / "batch_report_index.md", idx_md)
    idx_audit_md = render_markdown_index(report_items, title="lookupbooks_sys 批量问答索引（审查版）", meta=meta, mode="audit")
    write_markdown_report(batch_dir / "batch_report_audit_index.md", idx_audit_md)


def _resolve_book_output_dir(book_output_dir: str, outputs_root: Path) -> Path:
    """Resolve a user-provided book_output_dir.

    Accepts:
      - an existing path
      - a folder name under outputs_root
    """
    p = Path(book_output_dir)
    if p.exists():
        return p
    # try under outputs_root
    cand = outputs_root / book_output_dir
    if cand.exists():
        return cand
    cand2 = outputs_root / p.name
    if cand2.exists():
        return cand2
    return p



# -------------------------
# Mock LLM (for offline tests)
# Set env LOOKUPBOOKS_SYS_MOCK_LLM=1 to enable.
# -------------------------
class _MockRespMsg:
    def __init__(self, content: str):
        self.content = content

class _MockChoice:
    def __init__(self, content: str):
        self.message = _MockRespMsg(content)

class _MockResp:
    def __init__(self, content: str):
        self.choices = [_MockChoice(content)]

def _mock_llm_reply(messages: List[Dict[str, Any]]) -> str:
    text = "\n".join([str(m.get("content","")) for m in messages if isinstance(m, dict)])
    t = text
    # TRUNCATION_TEST: used by tools/self_check.py to ensure integrity guard works
    if "TRUNCATION_TEST" in t and "答案判定器" in t:
        return "B"
    if "TRUNCATION_TEST" in t and ("只返回A或B" in t or "只返回 A 或 B" in t):
        return "B"
    if "最终输出格式修复器" in t or "未通过最终格式校验" in t:
        return "【解题思路与公式推导思路】\n- （MOCK）格式已修复，保留原意。\n\n【最终答案】\n- （MOCK）最终结论。"
    if "TRUNCATION_TEST" in t and "请直接回答下面这个问题" in t:
        # intentionally broken (unclosed code fence + correction marker)
        return "【解题思路与公式推导思路】\n- （MOCK）这是一个故意截断的回答，用于测试完整性保护。\n\n【最终答案】\n```python\n# Correction: recall V = V0/\n"
    if "TRUNCATION_TEST" in t and "根据下面的知识回答问题" in t:
        return "【解题思路与公式推导思路】\n- （MOCK）完整推导要点。\n\n【最终答案】\n- （MOCK）最终结论：$a=1$, $b=2$。\n"

    # structured Step0 JSON
    if '"verdict": "A|B|C"' in t and 'should_use_tools' in t and 'direct_sufficient' in t:
        return '{"verdict":"A","direct_sufficient":true,"should_use_tools":false,"confidence":0.96,"rationale":"（MOCK）直接作答已足够。","missing_elements":[],"explicit_errors":[],"uncertainties":[]}'
    # M5 self-assess JSON
    if '"p_solve": 0.0' in t and '简短理由' in t:
        return '{"p_solve": 0.92, "reason": "（MOCK）可直接完成。"}'
    # judge JSON
    if 'schema_version' in t and 'used_evidence_indices' in t and 'unsupported_claims' in t and 'strengths' in t:
        qid = 'Q00'
        ms = re.findall(r'\"qid\"\s*:\s*\"([^\"]+)\"', t)
        if ms:
            qid = ms[-1]
        mode = ''
        ms2 = re.findall(r'\"mode\"\s*:\s*\"([^\"]+)\"', t)
        if ms2:
            mode = ms2[-1]
        return json.dumps({
            "schema_version": "judge_v1",
            "qid": qid,
            "mode": mode,
            "scores": {
                "overall": 8, "correctness": 8, "completeness": 8, "derivation": 8,
                "clarity": 8, "grounding": 8, "hallucination_resistance": 8
            },
            "should_use_tools": (mode in {"M2", "M3", "M5"}),
            "confidence": 0.88,
            "used_evidence_indices": [1],
            "unsupported_claims": [],
            "key_issues": [],
            "strengths": ["（MOCK）结构完整", "（MOCK）答案可解析"]
        }, ensure_ascii=False)

    # section-aware ranking JSON
    if "输出 ONLY JSON" in t and '"candidates":[book_id' in t:
        return '{"candidates":[1],"rationales":{"1":"（MOCK）候选书。"}}'
    if "输出 ONLY JSON" in t and '"selected":[book_id' in t:
        return '{"selected":[1],"why":{"1":"（MOCK）选书。"}}'
    if "输出 ONLY JSON" in t and '"candidates":[[book_id, chapter_no]' in t:
        return '{"candidates":[[1,1]],"why":{"1:1":"（MOCK）候选章。"}}'
    if "输出 ONLY JSON" in t and '"selected":[[book_id, chapter_no]' in t:
        return '{"selected":[[1,1]],"why":{"1:1":"（MOCK）选章。"}}'
    if "输出 ONLY JSON" in t and '"selected":[[book_id, chapter_no, section_id]' in t:
        return '{"selected":[[1,1,"s1"]],"why":{"1:1:s1":"（MOCK）选节。"}}'
    # verdicts
    if "A/B/C" in t or "返回A、B、C" in t or "返回 A、B、C" in t:
        return "A"
    if "只返回A或B" in t or "只返回 A 或 B" in t:
        return "A"
    # generic summarize/answer
    if "100~200" in t and "介绍" in t:
        return "（MOCK）本书主要覆盖与问题相关的核心概念与方法，适合作为快速入门与复习参考。"
    if "每一章" in t and "100~200" in t:
        return "1\tChapter 1\t（MOCK）该章介绍核心概念与基本公式。\n"
    # post-format JSON
    
    if "Markdown/LaTeX排版规范化器" in t and "final_reasoning" in t and "final_answer" in t and "audit_reasoning" in t:
        return "{\"final_reasoning\":\"（MOCK）关键推导要点。\",\"final_answer\":\"（MOCK）最终结论。\",\"audit_reasoning\":\"（MOCK）更完整的推导过程（用于审查）。\"}"
    if "final_reasoning" in t and "audit_reasoning" in t and "ONLY JSON" in t:
        return "{\"final_reasoning\":\"（MOCK）关键推导要点。\",\"final_answer\":\"（MOCK）最终结论。\",\"audit_reasoning\":\"（MOCK）更完整的推导过程（用于审查）。\"}"
    # default
    return "（MOCK）已生成回答（用于离线自检）。"

class _MockCompletions:
    def create(self, **kwargs):
        messages = kwargs.get("messages") or []
        return _MockResp(_mock_llm_reply(messages))

class _MockChat:
    def __init__(self):
        self.completions = _MockCompletions()

class _MockClient:
    def __init__(self):
        self.chat = _MockChat()

class _MockSession:
    def __init__(self):
        self.client = _MockClient()
        self.max_retries = 0
        self.backoff_s = [0.0]
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

class _MockApiPool:
    def __init__(self):
        self.keys = ["MOCK_KEY"]
    def session(self):
        return _MockSession()

def _build_pool(cfg) -> ApiPool:
    if os.environ.get("LOOKUPBOOKS_SYS_MOCK_LLM", "").strip() == "1":
        return _MockApiPool()
    # default key pool: supports configurable env prefix/count; fallback to historical QWEN_API_KEY_*.
    key_env_prefix = str(getattr(cfg.models, "key_env_prefix", "QWEN_API_KEY_") or "QWEN_API_KEY_")
    key_env_count = int(getattr(cfg.models, "key_env_count", 10) or 10)
    keys = env_keys(key_env_prefix, key_env_count)
    if not keys and key_env_prefix != "QWEN_API_KEY_":
        keys = env_keys("QWEN_API_KEY_", key_env_count)
    if not keys:
        raise SystemExit(
            f"No keys found. Set {key_env_prefix}0..{key_env_prefix}{key_env_count-1} (recommended) or 1-based variant in environment."
        )
    max_retries = int(getattr(cfg.models, "max_retries", 0) or 0)
    backoff_s = list(getattr(cfg.models, "backoff_s", [2, 4, 8]) or [2, 4, 8])
    return ApiPool(
        keys,
        base_url=str(cfg.models.base_url),
        timeout_s=int(cfg.models.request_timeout_s),
        max_retries=max_retries,
        backoff_s=backoff_s,
    )


def _load_or_import_library(outputs_root: Path, library_root: Path, *, overwrite: bool = False) -> Tuple[BookLibrary, List[BookIndexEntry]]:
    """Return (BookLibrary, entries). Import from outputs_root if index is empty."""
    ensure_dir(library_root)
    lib = BookLibrary(library_root)
    entries = lib.load_index()
    if not entries:
        import_from_pdf_txt_align_outputs(outputs_root, library_root, overwrite=overwrite)
        entries = lib.load_index()
    return lib, entries


def _select_model(cfg, key: str, default: str) -> str:
    try:
        qa_models = getattr(cfg, "qa_models")
        v = getattr(qa_models, key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    except Exception:
        pass
    return default


def _ensure_summaries(
    session,
    cfg,
    lib: BookLibrary,
    entries: List[BookIndexEntry],
    *,
    summarybook: bool = True,
    chapter_summaries: bool = True,
    only_book_ids: Optional[List[int]] = None,
    skip_existing: bool = False,
) -> None:
    """Ensure summarybook.txt and per-book chapter summaries exist.

    NOTE (v0.4.2+): We generate chapter summaries FIRST, then generate summarybook.
    Rationale: summarybook prefers using already-generated chapter summaries as its
    source; generating in the opposite order makes summarybook stale until a
    second summarize pass.
    """
    llm_default = str(getattr(getattr(cfg, "models"), "llm_model"))
    summarize_model = _select_model(cfg, "summarize_model", llm_default)

    # 1) Chapter summaries first (so summarybook can reuse them)
    if chapter_summaries:
        target = entries
        if only_book_ids:
            ids = {int(x) for x in only_book_ids}
            target = [e for e in entries if e.book_id in ids]

        for e in target:
            outp = lib.chapter_summary_path(e.book_id)
            if skip_existing and outp.exists():
                continue
            build_chapter_summaries_for_book(session, model=summarize_model, library=lib, book=e)

    # 2) Summarybook after chapter summaries
    if summarybook:
        outp = lib.summarybook_path()
        if (not skip_existing) or (not outp.exists()):
            build_book_summaries(
                session,
                model=summarize_model,
                library=lib,
                entries=entries,
                out_path=outp,
            )
def _parse_q0_blocks(md_text: str, tag: str = "q0") -> List[Dict[str, Any]]:
    """Parse Markdown blocks like ```q0 ... ``` and nearest heading '## Q01｜title'."""
    heading_pat = re.compile(r"^##\s*(Q\d+)\s*[｜|]\s*(.+?)\s*$", re.MULTILINE)
    headings = [(m.start(), m.group(1), m.group(2).strip()) for m in heading_pat.finditer(md_text)]

    code_pat = re.compile(rf"```{re.escape(tag)}\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
    blocks = [(m.start(), m.end(), m.group(1).strip()) for m in code_pat.finditer(md_text)]

    items: List[Dict[str, Any]] = []
    for i, (s, _e, qtext) in enumerate(blocks, start=1):
        qid = f"Q{i:02d}"
        title = f"Question {i}"
        prev = [h for h in headings if h[0] < s]
        if prev:
            _pos, qid_found, title_found = prev[-1]
            qid, title = qid_found, title_found
        items.append({"idx": i, "qid": qid, "title": title, "question": qtext})
    return items


# -------------------------
# Commands
# -------------------------
def cmd_list(args) -> None:
    pairs = find_pairs(Path(args.pairs_dir))
    for pdf, txt in pairs:
        print(f"{pdf.name}  <->  {txt.name}")
    print(f"Total pairs: {len(pairs)}")


def cmd_align(args) -> None:
    cfg = load_config(args.config)
    out_root = ensure_dir(Path(cfg.runtime.out_dir))

    if args.pairs_dir:
        pairs = find_pairs(Path(args.pairs_dir))
    elif args.pdf and args.text:
        pairs = [(Path(args.pdf), Path(args.text))]
    else:
        raise SystemExit("Provide --pairs-dir OR (--pdf and --text).")

    if not pairs:
        raise SystemExit("No pairs found.")

    pool = _build_pool(cfg)
    keys = list(pool.keys)
    workers = int(args.workers) if args.workers is not None else min(len(keys), max(1, len(pairs)))
    workers = max(1, workers)
    if workers > len(keys):
        print(f"[WARN] workers={workers} > api_keys={len(keys)}; extra workers will block on ApiPool.")

    run_id = now_ts()
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        config_hash = sha1_bytes(Path(args.config).read_bytes())
    except Exception:
        config_hash = ""

    project_root = Path(__file__).resolve().parent
    try:
        from pdf_txt_align.utils import compute_code_hash
        code_hash = compute_code_hash(project_root)[:12]
    except Exception:
        code_hash = ""

    env_ver = os.environ.get("PDFTXTALIGN_CODE_VERSION")
    try:
        from pdf_txt_align import __version__ as pkg_version
    except Exception:
        pkg_version = None
    code_version_source = "env" if env_ver else "default"
    code_version = env_ver or (pkg_version or "unknown")

    # Propagate run metadata (useful for logs/debug)
    os.environ["PDFTXTALIGN_RUN_ID"] = str(run_id)
    os.environ["PDFTXTALIGN_CONFIG_HASH"] = str(config_hash)
    os.environ["PDFTXTALIGN_CODE_HASH"] = str(code_hash)
    os.environ["PDFTXTALIGN_CODE_VERSION"] = str(code_version)
    os.environ["PDFTXTALIGN_CODE_VERSION_SOURCE"] = str(code_version_source)
    os.environ["PDFTXTALIGN_CONFIG_PATH"] = str(Path(args.config).resolve())
    os.environ["PDFTXTALIGN_CWD"] = str(Path.cwd())
    os.environ["PDFTXTALIGN_PID"] = str(os.getpid())

    batch_dir = ensure_dir(out_root / "_batch" / run_id)
    results: List[Dict[str, Any]] = []

    def _one(pdf: Path, txt: Path) -> Dict[str, Any]:
        with pool.session() as session:
            return process_one_book(session, Path(args.config), pdf, txt)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, pdf, txt) for pdf, txt in pairs]
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"success": False, "fallbacks": ["exception"], "error": str(e), "traceback": traceback.format_exc()})

    dump_json(batch_dir / "batch_summary.json", {
        "run_id": run_id,
        "started_at": started_at,
        "code_version": code_version,
        "code_version_source": code_version_source,
        "code_hash": code_hash,
        "config_hash": config_hash,
        "config_path": str(Path(args.config).resolve()),
        "workers": workers,
        "pairs": len(pairs),
        "results": results,
    })

    fb_csv = batch_dir / "fallback_books.csv"
    with fb_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["book_id", "success", "fallbacks", "num_chapters", "out_dir"])
        for r in results:
            out_dir = Path(str(r.get("out_dir") or ""))
            ch_dir = out_dir / "chapters"
            num_ch = sum(1 for p in ch_dir.glob("*.txt") if p.is_file()) if ch_dir.exists() else 0
            w.writerow([
                r.get("book_id", ""),
                bool(r.get("success", False)),
                "|".join(r.get("fallbacks", []) or []),
                num_ch,
                str(out_dir),
            ])

    from collections import Counter
    c = Counter()
    for r in results:
        for fb in (r.get("fallbacks") or []):
            c[str(fb)] += 1
    dump_json(batch_dir / "batch_metrics.json", {"run_id": run_id, "fallback_counts": dict(c)})

    print(f"[OK] align done. outputs={out_root.resolve()} batch={batch_dir.resolve()}")


def cmd_build_library(args) -> None:
    outputs_root = Path(args.outputs_root)
    library_root = Path(args.library_root)
    import_from_pdf_txt_align_outputs(
        outputs_root,
        library_root,
        overwrite=bool(args.overwrite),
        mode=str(args.mode),
        start_book_id=int(args.start_book_id),
    )
    lib = BookLibrary(library_root)
    entries = lib.load_index()
    print(f"[OK] library imported: {library_root.resolve()}")
    print(f" - books indexed: {len(entries)}")

    # Always write/update registry for operator visibility.
    reg_path = sync_registry(library_root)
    print(f"[OK] registry updated: {reg_path.resolve()}")

    # In v0.4.2+, library build includes summary generation by default (v0.4.2 also supports summary refresh on add/remove).
    if bool(getattr(args, "skip_summarize", False)):
        print("NOTE: summaries skipped (--skip-summarize). Run `python run.py summarize` later if needed.")
        return

    cfg = load_config(args.agent_config)
    pool = _build_pool(cfg)
    with pool.session() as session:
        _ensure_summaries(session, cfg, lib, entries, summarybook=True, chapter_summaries=True, skip_existing=False)

    print(f"[OK] summaries generated under: {library_root.resolve()}")

    # Re-sync registry to reflect summary completion.
    reg_path = sync_registry(library_root)
    print(f"[OK] registry updated: {reg_path.resolve()}")


def cmd_library_list(args) -> None:
    library_root = Path(args.library_root)
    # Ensure registry is up-to-date, then display it.
    reg_path = sync_registry(library_root)
    rows, meta = read_registry(library_root)
    if not rows:
        print("(empty)")
        return
    print(f"[registry] {reg_path.resolve()}")
    print("book_id\ttitle\tchapters\tchapter_summary_ok\tsource_dir\tnotes")
    for bid in sorted(rows.keys()):
        r = rows[bid]
        print(
            f"{r.book_id}\t{r.title}\t{r.chapters_count}\t{1 if r.chapter_summary_ok else 0}\t{r.source_dir}\t{r.notes}"
        )
    print(f"Total: {len(rows)}")


def cmd_library_add(args) -> None:
    library_root = Path(args.library_root)
    outputs_root = Path(getattr(args, "outputs_root", "./outputs"))

    # Pre-check: sync registry so operator can verify current state.
    sync_registry(library_root)

    # Resolve book_output_dir: allow passing either a full path or a folder name under outputs_root.
    raw_dir = str(args.book_output_dir)
    book_dir = _resolve_book_output_dir(raw_dir, outputs_root)
    if not book_dir.exists():
        suggestions = _suggest_similar_dirs(outputs_root, Path(raw_dir).name)
        msg = [
            f"book_output_dir not found: {book_dir}",
            f"Hint: available book folders under {outputs_root.resolve()} (with chapters/ or sections/) include:",
            *[f"  - {s}" for s in suggestions],
            "\nUse: python run.py library-add --book-output-dir \".\\outputs\\<folder_name>\" ...",
        ]
        raise FileNotFoundError("\n".join(msg))

    ent = add_book_from_output_dir(
        book_dir,
        library_root,
        book_id=args.book_id,
        title=args.title,
        mode=str(args.mode),
        overwrite=bool(args.overwrite),
    )
    print(f"[OK] added book: id={ent.book_id} title={ent.title}")

    # Auto refresh summaries by default (v0.4.2)
    if bool(getattr(args, "skip_refresh_summaries", False)):
        print("NOTE: summary refresh skipped (--skip-refresh-summaries). Run `python run.py summarize` later if needed.")
        return

    cfg = load_config(getattr(args, "agent_config", "./config/agent.yaml"))
    pool = _build_pool(cfg)
    lib = BookLibrary(library_root)
    entries = lib.load_index()

    with pool.session() as session:
        # Refresh summarybook for the whole library, and generate chapter summary for the newly added book only.
        _ensure_summaries(
            session,
            cfg,
            lib,
            entries,
            summarybook=True,
            chapter_summaries=True,
            only_book_ids=[int(ent.book_id)],
            skip_existing=False,
        )

    print(f"[OK] summaries refreshed: summarybook + book{ent.book_id}_chapter_summary.txt")

    # Post-check: sync registry to reflect changes.
    reg_path = sync_registry(library_root)
    print(f"[OK] registry updated: {reg_path.resolve()}")


def cmd_library_remove(args) -> None:
    library_root = Path(args.library_root)
    book_id = int(args.book_id)

    # Pre-check: sync registry and confirm the target exists.
    sync_registry(library_root)
    rows, _ = read_registry(library_root)
    if book_id not in rows:
        print(f"[WARN] book_id not found in registry: {book_id}")
        if rows:
            print("Existing book_ids:", ", ".join(str(x) for x in sorted(rows.keys())))
        return

    ok = remove_book_from_library(library_root, book_id, delete_files=not bool(args.keep_files))
    if not ok:
        print(f"[WARN] book_id not found: {book_id}")
        return

    print(f"[OK] removed book_id={book_id}")

    # Auto refresh summaries by default (v0.4.2)
    if bool(getattr(args, "skip_refresh_summaries", False)):
        print("NOTE: summary refresh skipped (--skip-refresh-summaries). Run `python run.py summarize` later if needed.")
        return

    cfg = load_config(getattr(args, "agent_config", "./config/agent.yaml"))
    pool = _build_pool(cfg)
    lib = BookLibrary(library_root)
    entries = lib.load_index()

    with pool.session() as session:
        # Removed book's chapter summary file is deleted by remove_book_from_library().
        # Refresh summarybook to reflect current index.
        _ensure_summaries(
            session,
            cfg,
            lib,
            entries,
            summarybook=True,
            chapter_summaries=False,
            skip_existing=False,
        )

    print(f"[OK] summarybook refreshed: {lib.summarybook_path()}")

    # Post-check: sync registry to reflect changes.
    reg_path = sync_registry(library_root)
    print(f"[OK] registry updated: {reg_path.resolve()}")



def cmd_summarize(args) -> None:
    cfg = load_config(args.agent_config)
    outputs_root = Path(args.outputs_root)
    library_root = Path(args.library_root)

    lib, entries = _load_or_import_library(outputs_root, library_root, overwrite=False)
    if not entries:
        raise SystemExit("No books in library. Run align/build-library or library-add first.")

    pool = _build_pool(cfg)

    with pool.session() as session:
        _ensure_summaries(
            session,
            cfg,
            lib,
            entries,
            summarybook=not bool(args.chapter_only),
            chapter_summaries=not bool(args.summarybook_only),
            only_book_ids=args.only_book_id,
        )

    print(f"[OK] summarize done: {library_root.resolve()}")
    reg_path = sync_registry(library_root)
    print(f"[OK] registry updated: {reg_path.resolve()}")


def _read_question(args) -> str:
    q = (args.question or "").strip()
    if args.question_file:
        qp = Path(args.question_file)
        if not qp.exists():
            raise SystemExit(f"question file not found: {qp}")
        q = qp.read_text(encoding="utf-8").strip()
    if not q:
        raise SystemExit("ask requires -q/--question OR --question-file")
    return q


def cmd_ask(args) -> None:
    q = _read_question(args)

    cfg = load_config(args.agent_config)
    outputs_root = Path(args.outputs_root)
    library_root = Path(args.library_root)

    lib, entries = _load_or_import_library(outputs_root, library_root, overwrite=False)
    if not entries:
        raise SystemExit("No books in library. Run align/build-library or library-add first.")

    pool = _build_pool(cfg)

    # Ensure summary files exist if requested
    if bool(args.auto_summarize) and (not lib.summarybook_path().exists()):
        with pool.session() as sess:
            _ensure_summaries(sess, cfg, lib, entries, summarybook=True, chapter_summaries=True, skip_existing=True)
    else:
        if not lib.summarybook_path().exists():
            raise SystemExit("summarybook.txt not found. Run `python run.py summarize` (or pass --auto-summarize).")

    agent = MultiBookQAAgent(lib, cfg)

    run_root = ensure_dir(Path(args.runs_dir))
    run_id = now_ts()
    out_dir = ensure_dir(run_root / run_id)

    with pool.session() as session:
        res = agent.ask(
            session,
            q,
            out_dir=out_dir,
            verbose=True,
            mode=getattr(args, "test_mode", None),
            topk_books_override=getattr(args, "topk_books", None),
            topk_pairs_override=getattr(args, "topk_pairs", None),
            topk_sections_override=getattr(args, "topk_sections", None),
        )

    out_json = out_dir / "qa_result.json"
    if out_json.exists():
        payload = json.loads(out_json.read_text(encoding="utf-8"))
    else:
        payload = res.__dict__
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== FINAL ANSWER ===\n")
    print(payload.get("answer") or payload.get("final_answer") or "")

    print("\n[Saved] " + str(out_json.resolve()))

    # Generate Markdown report for this run
    try:
        qid = str(getattr(args, "qid", "") or "Q00")
        title = str(getattr(args, "title", "") or "单题")
        item = build_item_from_qa_result(qid=qid, title=title, question=q, qa_result_payload=payload, q_dir=out_dir)
        md = render_markdown_report(
            [item],
            title="lookupbooks_sys 单题问答报告",
            meta={
                "run_id": run_id,
                "library_root": str(library_root.resolve()),
                "test_mode": str(payload.get("test_mode") or getattr(args, "test_mode", "") or ""),
            },
        )
        report_path = out_dir / "report.md"
        write_markdown_report(report_path, md)
        print("[Saved] " + str(report_path.resolve()))

        md_audit = render_markdown_report(
            [item],
            title="lookupbooks_sys 单题问答报告（思考过程审查版）",
            meta={
                "run_id": run_id,
                "library_root": str(library_root.resolve()),
                "test_mode": str(payload.get("test_mode") or getattr(args, "test_mode", "") or ""),
            },
            mode="audit",
        )
        report_audit_path = out_dir / "report_audit.md"
        write_markdown_report(report_audit_path, md_audit)
        print("[Saved] " + str(report_audit_path.resolve()))
    except Exception as e:
        print(f"[WARN] failed to write report.md: {e}")

    if bool(args.emit_json):
        print("\n=== JSON OUTPUT ===\n")
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_batch_ask(args) -> None:
    # Suite mode: orchestrate M1..M9 (and M8 k-sweep) without changing default batch-ask behavior.
    if str(getattr(args, 'suite', '') or '').strip():
        from lookupbooks_sys.suite_runner import run_batch_suite
        return run_batch_suite(args, run_py_path=Path(__file__).resolve())

    cfg = load_config(args.agent_config)
    outputs_root = Path(args.outputs_root)
    library_root = Path(args.library_root)
    batch_file = Path(args.batch_file)

    if not batch_file.exists():
        raise SystemExit(f"batch file not found: {batch_file}")

    md_text = batch_file.read_text(encoding="utf-8")
    items = _parse_q0_blocks(md_text, tag=str(args.tag))
    if not items:
        raise SystemExit(f"No ```{args.tag} ...``` blocks found in {batch_file}")

    start = int(args.start)
    end = int(args.end) if int(args.end) > 0 else len(items)
    selected = [x for x in items if start <= int(x["idx"]) <= end]

    lib, entries = _load_or_import_library(outputs_root, library_root, overwrite=False)
    if not entries:
        raise SystemExit("No books in library. Run align/build-library or library-add first.")

    pool = _build_pool(cfg)

    # Ensure summary files exist if requested
    if bool(args.auto_summarize) and (not lib.summarybook_path().exists()):
        with pool.session() as sess:
            _ensure_summaries(sess, cfg, lib, entries, summarybook=True, chapter_summaries=True, skip_existing=True)
    else:
        if not lib.summarybook_path().exists():
            raise SystemExit("summarybook.txt not found. Run `python run.py summarize` (or pass --auto-summarize).")

    batch_root = ensure_dir(Path(args.runs_dir))
    # Optional deterministic batch id (useful for wrappers / reproducible experiments)
    batch_id = str(getattr(args, "batch_id", "") or "").strip() or now_ts()
    batch_dir = ensure_dir(batch_root / batch_id)

    existing_qmeta = _load_batch_question_list(batch_dir)
    merged_qmeta = _merge_question_lists(existing_qmeta, selected)
    (batch_dir / "parsed_questions.json").write_text(json.dumps(merged_qmeta, ensure_ascii=False, indent=2), encoding="utf-8")

    existing_summary = _load_batch_summary(batch_dir)
    existing_results_by_qid = _summary_results_by_qid(existing_summary)

    agent = MultiBookQAAgent(lib, cfg)

    workers = int(args.workers) if args.workers is not None else min(len(pool.keys), max(1, len(selected)))
    workers = max(1, min(workers, len(pool.keys)))
    print(f"[BATCH] api_keys={len(pool.keys)} workers={workers} questions={len(selected)}", flush=True)

    summary: Dict[str, Any] = {
        "batch_id": batch_id,
        "batch_file": str(existing_summary.get("batch_file") or batch_file.resolve()),
        "started_at": str(existing_summary.get("started_at") or datetime.datetime.now().isoformat(timespec="seconds")),
        "items_total": len(merged_qmeta),
        "workers": workers,
        "results": _ordered_results_from_qmeta(merged_qmeta, existing_results_by_qid),
        "rerun_selected": len(selected),
    }

    def _one(item: Dict[str, Any]) -> Dict[str, Any]:
        qid = str(item["qid"])
        title = str(item.get("title") or "")
        qtext = str(item.get("question") or "").strip()
        q_dir = ensure_dir(batch_dir / qid)
        (q_dir / "q0.txt").write_text(qtext, encoding="utf-8")
        t0 = time.perf_counter()

        def _cleanup_retry_outputs() -> None:
            for name in ["qa_result.json", "report.md", "report_audit.md", "report_error.txt", "error.txt"]:
                p = q_dir / name
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass

        def _run_attempt() -> Dict[str, Any]:
            with pool.session() as session:
                _ = agent.ask(
                    session,
                    qtext,
                    out_dir=q_dir,
                    verbose=bool(args.verbose),
                    mode=getattr(args, "test_mode", None),
                    topk_books_override=getattr(args, "topk_books", None),
                    topk_pairs_override=getattr(args, "topk_pairs", None),
                    topk_sections_override=getattr(args, "topk_sections", None),
                )
            out_json = q_dir / "qa_result.json"

            # Per-question markdown reports (helps when batch_report.md is too large to preview).
            if out_json.exists():
                payload = json.loads(out_json.read_text(encoding="utf-8"))
                try:
                    item_r = build_item_from_qa_result(qid=qid, title=title, question=qtext, qa_result_payload=payload, q_dir=q_dir)
                    md1 = render_markdown_report(
                        [item_r],
                        title=f"lookupbooks_sys 批量问答单题报告（{qid}）",
                        meta={
                            "batch_id": batch_id,
                            "batch_file": str(batch_file.resolve()),
                            "library_root": str(library_root.resolve()),
                            "test_mode": str(payload.get("test_mode") or getattr(args, "test_mode", "") or ""),
                        },
                    )
                    write_markdown_report(q_dir / "report.md", md1)

                    md2 = render_markdown_report(
                        [item_r],
                        title=f"lookupbooks_sys 批量问答单题报告（思考过程审查版｜{qid}）",
                        meta={
                            "batch_id": batch_id,
                            "batch_file": str(batch_file.resolve()),
                            "library_root": str(library_root.resolve()),
                            "test_mode": str(payload.get("test_mode") or getattr(args, "test_mode", "") or ""),
                        },
                        mode="audit",
                    )
                    write_markdown_report(q_dir / "report_audit.md", md2)
                except Exception:
                    # do not fail the whole question if report rendering fails
                    (q_dir / "report_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            return {
                "qid": qid,
                "title": title,
                "status": "ok",
                "mode": str(getattr(args, "test_mode", "") or ""),
                "duration_sec": round(float(time.perf_counter() - t0), 6),
                "qa_result": str(out_json.resolve()) if out_json.exists() else None,
                "attempts": 1,
            }

        max_attempts = max(1, int(getattr(args, "max_attempts", 3) or 3))
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    _cleanup_retry_outputs()
                result = _run_attempt()
                result["attempts"] = attempt
                if attempt > 1:
                    print(f"[BATCH] RETRY-SUCCESS {qid} attempt={attempt}", flush=True)
                return result
            except Exception as e:
                retry_log = q_dir / "retry.log"
                with retry_log.open("a", encoding="utf-8") as f:
                    f.write(f"[attempt={attempt}] error={e}\n")
                    f.write(traceback.format_exc())
                    f.write("\n---\n")
                if attempt < max_attempts:
                    print(f"[BATCH] RETRY {qid} after error={e}", flush=True)
                    time.sleep(min(5.0, 1.5 * attempt))
                    continue
                (q_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                return {
                    "qid": qid,
                    "title": title,
                    "status": "failed",
                    "mode": str(getattr(args, "test_mode", "") or ""),
                    "duration_sec": round(float(time.perf_counter() - t0), 6),
                    "error": str(e),
                    "traceback": str((q_dir / "error.txt").resolve()),
                    "attempts": attempt,
                }

    print(f"[BATCH] Start. Questions={len(selected)} workers={workers}", flush=True)
    print(f"[BATCH] Output dir: {batch_dir.resolve()}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, it) for it in selected]
        for fut in as_completed(futs):
            r = fut.result()
            existing_results_by_qid[str(r.get("qid") or "").strip()] = r
            summary["results"] = _ordered_results_from_qmeta(merged_qmeta, existing_results_by_qid)
            (batch_dir / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            if r.get("status") != "ok":
                print(f"[BATCH] FAILED {r.get('qid')} err={r.get('error')} attempts={r.get('attempts',1)}", flush=True)
                if bool(args.stop_on_error):
                    break
            else:
                print(f"[BATCH] OK {r.get('qid')} attempts={r.get('attempts',1)}", flush=True)

    summary["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    summary["results"] = _ordered_results_from_qmeta(merged_qmeta, existing_results_by_qid)
    ok_cnt = sum(1 for r in summary["results"] if r.get("status") == "ok")
    summary["ok_count"] = ok_cnt
    summary["fail_count"] = len(summary["results"]) - ok_cnt
    (batch_dir / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[BATCH] Done. OK={summary['ok_count']} FAIL={summary['fail_count']}", flush=True)
    print(f"[BATCH] Summary: {(batch_dir / 'batch_summary.json').resolve()}")

    # Generate Markdown report for the whole batch
    try:
        report_items: List[ReportItem] = []
        for it in sorted(merged_qmeta, key=lambda x: int(x.get("idx", 0))):
            qid = str(it.get("qid") or "")
            title = str(it.get("title") or "")
            qtext = str(it.get("question") or "")
            q_dir = batch_dir / qid
            out_json = q_dir / "qa_result.json"
            if out_json.exists():
                payload = json.loads(out_json.read_text(encoding="utf-8"))
                report_items.append(build_item_from_qa_result(qid=qid, title=title, question=qtext, qa_result_payload=payload, q_dir=q_dir))
            else:
                err = "qa_result.json missing"
                errp = q_dir / "error.txt"
                if errp.exists():
                    err = errp.read_text(encoding="utf-8", errors="ignore")[:1000]
                report_items.append(ReportItem(qid=qid, title=title, question=qtext, outline="", audit_outline="", final_answer="", status="failed", error=err))

        md = render_markdown_report(
            report_items,
            title="lookupbooks_sys 批量问答报告",
            meta={
                "batch_id": batch_id,
                "batch_file": str(batch_file.resolve()),
                "library_root": str(library_root.resolve()),
                "workers": str(workers),
                "test_mode": str(getattr(args, "test_mode", "") or ""),
            },
        )
        report_path = batch_dir / "batch_report.md"
        write_markdown_report(report_path, md)
        print(f"[BATCH] Report: {report_path.resolve()}")

        md_audit = render_markdown_report(
            report_items,
            title="lookupbooks_sys 批量问答报告（思考过程审查版）",
            meta={
                "batch_id": batch_id,
                "batch_file": str(batch_file.resolve()),
                "library_root": str(library_root.resolve()),
                "workers": str(workers),
                "test_mode": str(getattr(args, "test_mode", "") or ""),
            },
            mode="audit",
        )
        report_audit_path = batch_dir / "batch_report_audit.md"
        write_markdown_report(report_audit_path, md_audit)
        print(f"[BATCH] Audit Report: {report_audit_path.resolve()}")

        # Lightweight index reports (recommended to read/preview first)
        try:
            idx_md = render_markdown_index(
                report_items,
                title="lookupbooks_sys 批量问答索引（推荐先看）",
                meta={
                    "batch_id": batch_id,
                    "batch_file": str(batch_file.resolve()),
                    "library_root": str(library_root.resolve()),
                    "workers": str(workers),
                    "test_mode": str(getattr(args, "test_mode", "") or ""),
                },
                mode="final",
                link_mode="relative",
                base_dir=batch_dir,
            )
            write_markdown_report(batch_dir / "batch_report_index.md", idx_md)

            idx_audit_md = render_markdown_index(
                report_items,
                title="lookupbooks_sys 批量问答索引（审查版｜推荐先看）",
                meta={
                    "batch_id": batch_id,
                    "batch_file": str(batch_file.resolve()),
                    "library_root": str(library_root.resolve()),
                    "workers": str(workers),
                    "test_mode": str(getattr(args, "test_mode", "") or ""),
                },
                mode="audit",
                link_mode="relative",
                base_dir=batch_dir,
            )
            write_markdown_report(batch_dir / "batch_report_audit_index.md", idx_audit_md)
            print(f"[BATCH] Index: {(batch_dir / 'batch_report_index.md').resolve()}")
        except Exception:
            print("[WARN] failed to write batch index reports")
    except Exception as e:
        print(f"[WARN] failed to write batch_report.md: {e}")



def cmd_refresh_reports(args) -> None:
    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        raise SystemExit(f"batch dir not found: {batch_dir}")
    bs = batch_dir / "batch_summary.json"
    summary = json.loads(bs.read_text(encoding="utf-8")) if bs.exists() else {}
    meta = {
        "batch_id": str(summary.get("batch_id") or batch_dir.name),
        "batch_file": str(summary.get("batch_file") or ""),
        "library_root": str(getattr(args, "library_root", "") or ""),
        "workers": str(summary.get("workers") or ""),
        "test_mode": str(getattr(args, "test_mode", "") or ""),
    }
    _refresh_batch_markdown_reports(
        batch_dir,
        batch_title="lookupbooks_sys 批量问答报告",
        batch_title_audit="lookupbooks_sys 批量问答报告（思考过程审查版）",
        meta=meta,
    )
    print('[REPORT] refreshed:', str((batch_dir / 'batch_report.md').resolve()))


def cmd_aggregate_repeats(args) -> None:
    batch_dirs = [Path(x) for x in (args.batch_dir or [])]
    if not batch_dirs and getattr(args, 'glob', ''):
        batch_dirs = sorted([p for p in Path().glob(str(args.glob)) if p.is_dir()])
    if len(batch_dirs) < 1:
        raise SystemExit('aggregate-repeats requires at least 1 batch dir or --glob')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from lookupbooks_sys.eval.repeats import aggregate_repeat_batches
    paths = aggregate_repeat_batches(batch_dirs, out_dir=out_dir)
    print('[EVAL] aggregate-repeats outputs:', {k: str(v) for k, v in paths.items()})


def cmd_eval_batch(args) -> None:
    # Compute evaluation artifacts for a batch directory:
    # - auto metrics (fully programmatic)
    # - optional judge scoring (LLM-as-a-judge)
    # - aggregate combined.csv + summary.json + eval_report.md
    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        raise SystemExit(f"batch dir not found: {batch_dir}")

    cfg = load_config(args.agent_config)

    # 1) auto metrics
    from lookupbooks_sys.eval.auto_metrics import run_auto_metrics
    paths_auto = run_auto_metrics(batch_dir=batch_dir)
    print("[EVAL] auto_metrics:", {k: str(v) for k, v in paths_auto.items()})

    # 2) optional judge
    if bool(getattr(args, 'judge', False)):
        eval_cfg = getattr(cfg, 'eval', None)
        rubric_path = Path(getattr(args, 'rubric', '') or getattr(eval_cfg, 'rubric_path', ''))
        if not rubric_path.exists():
            raise SystemExit(f"rubric file not found: {rubric_path}")
        pool = _build_pool(cfg)
        from lookupbooks_sys.eval.judge import run_batch_judge
        workers = int(getattr(args, 'judge_workers', 0) or getattr(eval_cfg, 'judge_workers', 1) or 1)
        workers = max(1, min(workers, len(pool.keys)))
        print(f"[EVAL-JUDGE] api_keys={len(pool.keys)} judge_workers={workers}", flush=True)
        paths_j = run_batch_judge(
            batch_dir=batch_dir,
            pool=pool,
            cfg=cfg,
            rubric_path=rubric_path,
            skip_existing=bool(getattr(args, 'skip_existing', False)),
            workers=workers,
        )
        print("[EVAL] judge_scores:", {k: str(v) for k, v in paths_j.items()})

    # 3) aggregate
    from lookupbooks_sys.eval.aggregate import aggregate_batch
    paths_ag = aggregate_batch(batch_dir)
    print("[EVAL] aggregate:", {k: str(v) for k, v in paths_ag.items()})

    # 4) refresh markdown reports so per-question and batch reports can include judge scores when available
    try:
        bs = batch_dir / "batch_summary.json"
        summary = json.loads(bs.read_text(encoding="utf-8")) if bs.exists() else {}
        meta = {
            "batch_id": str(summary.get("batch_id") or batch_dir.name),
            "batch_file": str(summary.get("batch_file") or ""),
            "library_root": str(getattr(args, "library_root", "") or ""),
            "workers": str(summary.get("workers") or getattr(args, "judge_workers", "") or ""),
            "test_mode": str(getattr(args, "test_mode", "") or ""),
        }
        _refresh_batch_markdown_reports(
            batch_dir,
            batch_title="lookupbooks_sys 批量问答报告",
            batch_title_audit="lookupbooks_sys 批量问答报告（思考过程审查版）",
            meta=meta,
        )
        print("[EVAL] refreshed markdown reports:", str((batch_dir / "batch_report.md").resolve()))
    except Exception as e:
        print(f"[WARN] failed to refresh markdown reports after eval: {e}")


def cmd_eval_compare(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_dirs = [Path(x) for x in (args.batch_dir or [])]
    if len(batch_dirs) < 2:
        raise SystemExit('eval-compare requires at least 2 --batch-dir')

    from lookupbooks_sys.eval.aggregate import aggregate_batch
    for bd in batch_dirs:
        comb = bd / '_eval' / 'combined.csv'
        if not comb.exists():
            # try to aggregate if auto metrics already exist
            aggregate_batch(bd)

    from lookupbooks_sys.eval.compare import compare_batches
    paths = compare_batches(batch_dirs, out_dir=out_dir, baseline_mode=str(args.baseline_mode or 'M2'))
    print('[EVAL] compare outputs:', {k: str(v) for k, v in paths.items()})


def cmd_config_dump(args) -> None:
    cfg = load_config(args.config)
    print(json.dumps(cfg, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="lookupbooks_testsys v0.5.0 — 基于 lookupbooks_sys v1.1.2 的书/章/节检索测试系统，支持 Step0/1/2/3/4/5/6 问答与 M1..M9 suite 测试。"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("align", help="运行章节切分流水线（PDF 为基准）")
    sp.add_argument("--config", default="./config/default.yaml", help="YAML config for splitting pipeline")
    sp.add_argument("--pairs-dir", default=None, help="Directory containing *.pdf and matching *.txt/*.tex by stem")
    sp.add_argument("--pdf", default=None, help="Single PDF path (use with --text)")
    sp.add_argument("--text", default=None, help="Single TXT/TeX path (use with --pdf)")
    sp.add_argument("--workers", type=int, default=None, help="Parallel workers (default=min(keys, pairs))")
    sp.set_defaults(func=cmd_align)

    sp = sub.add_parser("list", help="列出 pairs-dir 下可匹配的 (pdf, txt/tex) 对")
    sp.add_argument("--pairs-dir", required=True, help="pairs directory")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("build-library", help="把 outputs/*/chapters 导入为 note.docx 的 library 结构（全量重扫）")
    sp.add_argument("--outputs-root", default="./outputs", help="Root of split outputs")
    sp.add_argument("--library-root", default="./library", help="Library root")
    sp.add_argument("--agent-config", default="./config/agent.yaml", help="Agent YAML config (used for summary generation)")
    sp.add_argument("--skip-summarize", action="store_true", help="Skip generating summarybook/chapter summaries during build-library")
    sp.add_argument("--start-book-id", type=int, default=1, help="First book id to assign if library empty")
    sp.add_argument("--mode", choices=["copy", "move"], default="copy", help="copy or move chapters into library")
    sp.add_argument("--overwrite", action="store_true", help="Overwrite existing imported books with same title")
    sp.set_defaults(func=cmd_build_library)

    sp = sub.add_parser("summarize", help="生成 summarybook.txt 与各书章节摘要（任务一）")
    sp.add_argument("--agent-config", default="./config/agent.yaml", help="YAML config for QA agent")
    sp.add_argument("--outputs-root", default="./outputs", help="Root of split outputs")
    sp.add_argument("--library-root", default="./library", help="Library root")
    sp.add_argument("--only-book-id", action="append", type=int, default=None, help="Only generate chapter summaries for this book_id (repeatable)")
    sp.add_argument("--summarybook-only", action="store_true", help="Only (re)generate summarybook.txt")
    sp.add_argument("--chapter-only", action="store_true", help="Only (re)generate per-book chapter summaries")
    sp.set_defaults(func=cmd_summarize)

    sp = sub.add_parser("ask", help="按 lookupbooks_sys v1.1.2 的书/章/节链路回答单个问题（任务二）")
    sp.add_argument("--agent-config", default="./config/agent.yaml", help="YAML config for QA agent")
    sp.add_argument("--outputs-root", default="./outputs", help="Root of split outputs")
    sp.add_argument("--library-root", default="./library", help="Library root")
    sp.add_argument("-q", "--question", required=False, default=None, help="Question string (optional if --question-file is provided)")
    sp.add_argument("--question-file", required=False, default=None, help="Path to UTF-8 text file containing the question (recommended for long/multi-line Q0)")
    sp.add_argument("--runs-dir", default="./runs/ask", help="Directory to write qa_result.json")
    sp.add_argument("--qid", default="Q00", help="Optional question id for reporting (default: Q00)")
    sp.add_argument("--title", default="单题", help="Optional title for reporting (default: 单题)")
    sp.add_argument("--auto-summarize", action="store_true", help="Auto-generate summaries if missing (will call LLM)")
    sp.add_argument("--emit-json", action="store_true", help="Print qa_result.json content to stdout (for front-end integration)")
    sp.add_argument("--test-mode", default=None, help="Test mode: M1..M9 (overrides qa.test_mode)")
    sp.add_argument("--topk-books", type=int, default=None, help="Override Step1 top-k books (used by M8/k-sweep)")
    sp.add_argument("--topk-pairs", type=int, default=None, help="Legacy override for chapter-level budget (kept for compatibility)")
    sp.add_argument("--topk-sections", type=int, default=None, help="Override Step5 total selected sections (used by M8/k-sweep)")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("batch-ask", help="批量处理 Markdown 测试题集中的 ```q0 ...```（内置版）")
    sp.add_argument("--batch-file", required=True, help="Markdown test file containing ```q0 blocks")
    sp.add_argument("--tag", default="q0", help="Code block tag name (default: q0)")
    sp.add_argument("--agent-config", default="./config/agent.yaml", help="YAML config for QA agent")
    sp.add_argument("--outputs-root", default="./outputs", help="Root of split outputs")
    sp.add_argument("--library-root", default="./library", help="Library root")
    sp.add_argument("--runs-dir", default="./runs/batch", help="Directory to write batch results")
    sp.add_argument("--batch-id", default=None, help="Optional fixed batch_id (folder name under --runs-dir)")
    sp.add_argument("--workers", type=int, default=None, help="Parallel question workers (default=min(keys, questions))")
    sp.add_argument("--max-attempts", type=int, default=3, help="Per-question retry attempts for batch-ask (default: 3)")
    sp.add_argument("--start", type=int, default=1, help="Start question index (1-based)")
    sp.add_argument("--end", type=int, default=0, help="End question index inclusive (0 means all)")
    sp.add_argument("--stop-on-error", action="store_true", help="Stop on first failure")
    sp.add_argument("--auto-summarize", action="store_true", help="Auto-generate summaries if missing (will call LLM)")
    sp.add_argument("--verbose", action="store_true", help="Print per-step logs for each question")
    sp.add_argument("--test-mode", default=None, help="Test mode: M1..M9 (overrides qa.test_mode)")
    sp.add_argument("--topk-books", type=int, default=None, help="Override Step1 top-k books (used by M8/k-sweep)")
    sp.add_argument("--topk-pairs", type=int, default=None, help="Legacy override for chapter-level budget (kept for compatibility)")
    sp.add_argument("--topk-sections", type=int, default=None, help="Override Step5 total selected sections (used by M8/k-sweep)")

    # --- Suite mode (optional): orchestrate M1..M9 end-to-end pass testing ---
    sp.add_argument("--suite", default="", help="Run a test suite, e.g. 'M1-M9' or 'M2,M3,M5'. Empty=normal batch-ask.")
    sp.add_argument("--suite-id", default="smoke_testsys_small", help="Suite id (folder name under --suite-runs-dir)")
    sp.add_argument("--suite-runs-dir", default="./runs/suite", help="Root directory for suite artifacts")
    sp.add_argument("--suite-m8-ks", default="1,3,5", help="Comma-separated k list for M8 sweep (default: 1,3,5)")
    sp.add_argument("--suite-judge", action="store_true", help="Run judge scoring for each stage (calls eval-batch --judge)")
    sp.add_argument("--suite-rubric", default="", help="Override rubric path for suite judge (default=config.eval.rubric_path)")
    sp.add_argument("--suite-judge-workers", type=int, default=0, help="Override judge workers for suite (default=config.eval.judge_workers)")
    sp.add_argument("--suite-skip-existing-judge", action="store_true", help="Skip existing judge cache in suite (judge_outputs/<Qxxx>.json)")
    sp.add_argument("--suite-stop-on-error", action="store_true", help="Stop suite on first stage failure (default: continue)")
    sp.add_argument("--suite-baseline-mode", default="M2", help="Baseline mode used by eval-compare delta (default=M2)")
    sp.set_defaults(func=cmd_batch_ask)

    # --- Evaluation ---
    sp = sub.add_parser('refresh-reports', help='离线刷新单题与批次 Markdown 报告（不调用 API）')
    sp.add_argument('--batch-dir', required=True, help='Path to runs/batch/<batch_id> directory')
    sp.add_argument('--library-root', default='', help='Optional library root for report metadata only')
    sp.add_argument('--test-mode', default='', help='Optional test mode for report metadata only')
    sp.set_defaults(func=cmd_refresh_reports)

    sp = sub.add_parser('aggregate-repeats', help='聚合多轮 batch 结果，生成每题每模式跨轮平均表现')
    sp.add_argument('--out-dir', required=True, help='Output directory for repeat aggregation artifacts')
    sp.add_argument('--glob', default='', help='Optional glob pattern for batch dirs')
    sp.add_argument('--batch-dir', action='append', default=[], help='Repeatable. Path to runs/batch/<batch_id> directory')
    sp.set_defaults(func=cmd_aggregate_repeats)


    sp = sub.add_parser('eval-batch', help='对 runs/batch/<batch_id> 生成统计与评测（auto_metrics + 可选 judge + aggregate）')
    sp.add_argument('--batch-dir', required=True, help='Path to runs/batch/<batch_id> directory')
    sp.add_argument('--agent-config', default='./config/agent.yaml', help='YAML config (models + eval settings)')
    sp.add_argument('--judge', action='store_true', help='Enable LLM-as-a-judge scoring')
    sp.add_argument('--rubric', default='', help='Override rubric markdown path (default=config.eval.rubric_path)')
    sp.add_argument('--judge-workers', type=int, default=0, help='Override judge workers (default=config.eval.judge_workers)')
    sp.add_argument('--skip-existing', action='store_true', help='Skip questions with existing judge_outputs/<Qxxx>.json cache')
    sp.set_defaults(func=cmd_eval_batch)

    sp = sub.add_parser('eval-compare', help='对多个 batch_dir 的 combined.csv 做对比（生成 wide 表与 delta 表）')
    sp.add_argument('--out-dir', required=True, help='Output directory for comparison artifacts')
    sp.add_argument('--baseline-mode', default='M2', help='Baseline mode used for delta (default=M2)')
    sp.add_argument('--batch-dir', action='append', default=[], help='Repeatable. Path to runs/batch/<batch_id> directory')
    sp.set_defaults(func=cmd_eval_compare)

    sp = sub.add_parser('scan-batch-failures', help='检查多个 batch_dir 中失败的 answer/judge（不调用 API）')
    sp.add_argument('--out-dir', required=True, help='Output directory for failure scan artifacts')
    sp.add_argument('--glob', default='', help='Optional glob pattern for batch dirs')
    sp.add_argument('--batch-dir', action='append', default=[], help='Repeatable. Path to runs/batch/<batch_id> directory')
    sp.add_argument('--no-expect-judge', action='store_true', help='Only check answer failures, do not require judge outputs')
    sp.set_defaults(func=cmd_scan_batch_failures)

    sp = sub.add_parser('repair-batch-failures', help='删除并补跑多个 batch_dir 中失败的 answer/judge')
    sp.add_argument('--out-dir', required=True, help='Output directory for repair artifacts')
    sp.add_argument('--agent-config', default='./config/agent.yaml', help='YAML config for QA agent and judge')
    sp.add_argument('--library-root', required=True, help='Library root (recommend absolute path)')
    sp.add_argument('--outputs-root', default='./outputs', help='Root of split outputs')
    sp.add_argument('--workers', type=int, default=24, help='Parallel question workers for answer rerun')
    sp.add_argument('--judge-workers', type=int, default=24, help='Parallel judge workers for judge rerun')
    sp.add_argument('--max-attempts', type=int, default=3, help='Per-question retry attempts for answer rerun')
    sp.add_argument('--verbose', action='store_true', help='Verbose rerun logs')
    sp.add_argument('--glob', default='', help='Optional glob pattern for batch dirs')
    sp.add_argument('--batch-dir', action='append', default=[], help='Repeatable. Path to runs/batch/<batch_id> directory')
    sp.set_defaults(func=cmd_repair_batch_failures)


    sp = sub.add_parser("library-list", help="列出当前 library 中的书籍索引")
    sp.add_argument("--library-root", default="./library")
    sp.set_defaults(func=cmd_library_list)

    sp = sub.add_parser("library-add", help="把 outputs/<book>/ 目录增量加入 library（不做全量重扫）")
    sp.add_argument("--outputs-root", default="./outputs", help="Outputs root used to resolve --book-output-dir when a folder name is given")
    sp.add_argument(
        "--book-output-dir",
        required=True,
        help="Path to outputs/<book>/ directory containing chapters/ (or just the folder name under --outputs-root)",
    )
    sp.add_argument("--library-root", default="./library")
    sp.add_argument("--book-id", type=int, default=None, help="Optional fixed book_id (default: next available)")
    sp.add_argument("--title", default=None, help="Optional title override (default: directory name)")
    sp.add_argument("--mode", choices=["copy", "move"], default="copy")
    sp.add_argument("--overwrite", action="store_true", help="Overwrite existing chapter files if book_id exists")
    sp.add_argument("--agent-config", default="./config/agent.yaml", help="Agent YAML config (used for summary refresh)")
    sp.add_argument("--skip-refresh-summaries", action="store_true", help="Skip refreshing summarybook/chapter summaries after library-add")
    sp.set_defaults(func=cmd_library_add)

    sp = sub.add_parser("library-remove", help="从 library 删除一本书（并删除对应文件）")
    sp.add_argument("--library-root", default="./library")
    sp.add_argument("--book-id", required=True, type=int)
    sp.add_argument("--keep-files", action="store_true", help="Only remove from index; keep files on disk")
    sp.add_argument("--agent-config", default="./config/agent.yaml", help="Agent YAML config (used for summary refresh)")
    sp.add_argument("--skip-refresh-summaries", action="store_true", help="Skip refreshing summarybook after library-remove")
    sp.set_defaults(func=cmd_library_remove)

    sp = sub.add_parser("config-dump", help="打印某个 YAML 配置文件展开后的 JSON（用于调参排查）")
    sp.add_argument("--config", default="./config/agent.yaml")
    sp.set_defaults(func=cmd_config_dump)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
