# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schemas import AUTO_METRICS_SCHEMA_VERSION, auto_metrics_fieldnames


@dataclass
class BatchContext:
    batch_id: str
    batch_dir: Path
    q_map: Dict[str, Dict[str, Any]]
    summary_map: Dict[str, Dict[str, Any]]


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _safe_list(x) -> List[Any]:
    return x if isinstance(x, list) else []


def _safe_dict(x) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _first_trace(trace: List[Dict[str, Any]], step: str) -> Optional[Dict[str, Any]]:
    for t in trace or []:
        if isinstance(t, dict) and t.get("step") == step:
            return t
    return None


def _integrity_issues(s: str) -> List[str]:
    s = s or ""
    out: List[str] = []
    if s.count("```") % 2 == 1:
        out.append("unclosed_code_fence")
    if "Correction:" in s:
        out.append("correction_tail")
    if s.count("【解题思路与公式推导思路】") > 1:
        out.append("duplicate_reasoning_heading")
    if len(s.strip()) < 20:
        out.append("too_short")
    return out


def _is_severely_broken(issues: List[str]) -> bool:
    return any(x in issues for x in ["unclosed_code_fence", "correction_tail"])


def _classify_error(q_dir: Path, qa_path: Optional[Path]) -> str:
    if qa_path is None or not qa_path.exists():
        if (q_dir / "error.txt").exists():
            return "exception"
        return "missing_qa_result"
    if (q_dir / "report_error.txt").exists():
        return "report_error"
    return ""


def _check_spec(mode: str, row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    m = (mode or "").upper().strip()
    v: List[str] = []
    steps = row.get("executed_steps") or []
    retrieval = bool(row.get("retrieval_used"))
    s1 = int(row.get("step1_candidates_books_count") or 0)
    s2 = int(row.get("step2_selected_books_count") or 0)
    s3 = int(row.get("step3_candidate_chapters_count") or 0)
    s4 = int(row.get("step4_selected_chapters_count") or 0)
    s5 = int(row.get("step5_selected_sections_count") or 0)
    fs = int(row.get("final_sources_count") or 0)
    topk_sections = int(row.get("topk_sections") or 0)
    avail_sections = int(row.get("step5_available_sections_total") or 0)

    if "Step0" not in steps:
        v.append("missing_step0")

    if m == "M1":
        if not retrieval:
            v.append("M1_requires_retrieval")
        for s in ("Step1", "Step2", "Step3", "Step4", "Step5", "Step6"):
            if s not in steps:
                v.append(f"M1_missing_{s}")
        if s5 < 1:
            v.append("M1_requires_at_least_one_section")
        if fs < 1:
            v.append("M1_requires_final_sources")
    elif m == "M2":
        if retrieval:
            v.append("M2_forbids_retrieval")
        if any(x > 0 for x in [s1, s2, s3, s4, s5, fs]):
            v.append("M2_forbids_retrieval_outputs")
    elif m == "M3":
        if retrieval:
            for s in ("Step1", "Step2", "Step3", "Step4", "Step5", "Step6"):
                if s not in steps:
                    v.append(f"M3_retrieval_missing_{s}")
    elif m == "M4":
        if retrieval:
            v.append("M4_forbids_retrieval")
        if not bool(row.get("expert_prompt_used")):
            v.append("M4_requires_expert_prompt")
    elif m == "M5":
        ps = row.get("p_solve")
        th = row.get("confidence_threshold")
        if ps in (None, "") or th in (None, ""):
            v.append("M5_requires_p_solve_and_threshold")
        else:
            try:
                psf = float(ps)
                thf = float(th)
                gate = row.get("m5_gate_triggered_retrieval")
                if psf >= thf and retrieval:
                    v.append("M5_high_confidence_should_not_retrieve")
                if psf < thf and not retrieval:
                    v.append("M5_low_confidence_should_retrieve")
                if psf >= thf and gate not in (False, None):
                    v.append("M5_gate_flag_inconsistent_high")
                if psf < thf and gate not in (True, None):
                    v.append("M5_gate_flag_inconsistent_low")
            except Exception:
                v.append("M5_invalid_p_solve_or_threshold")
    elif m == "M6":
        if not retrieval:
            v.append("M6_requires_retrieval")
        if not bool(row.get("chapter_only_mode")):
            v.append("M6_requires_chapter_only_mode")
        for s in ("Step1", "Step2", "Step3", "Step4", "Step6"):
            if s not in steps:
                v.append(f"M6_missing_{s}")
        if "Step5" in steps:
            v.append("M6_should_skip_step5")
        if s4 < 1:
            v.append("M6_requires_selected_chapters")
        if s5 != 0:
            v.append("M6_should_not_select_sections")
    elif m == "M7":
        if not retrieval:
            v.append("M7_requires_retrieval")
        if not bool(row.get("skip_section_ranking")):
            v.append("M7_requires_skip_section_ranking")
        if row.get("step5_selection_mode") != "rule_based":
            v.append("M7_requires_rule_based_step5")
        if s5 < 1:
            v.append("M7_requires_selected_sections")
    elif m == "M8":
        if not retrieval:
            v.append("M8_requires_retrieval")
        for s in ("Step1", "Step2", "Step3", "Step4", "Step5", "Step6"):
            if s not in steps:
                v.append(f"M8_missing_{s}")
        if topk_sections <= 0:
            v.append("M8_requires_topk_sections")
        if avail_sections <= 0:
            v.append("M8_zero_available_sections")
        expected = min(topk_sections, avail_sections) if avail_sections > 0 else 0
        if s5 != expected:
            v.append("M8_section_budget_not_met")
        if s5 <= 0:
            v.append("M8_requires_at_least_one_section")
        if fs <= 0:
            v.append("M8_requires_final_sources")
    elif m == "M9":
        if not retrieval:
            v.append("M9_requires_retrieval")
        if not bool(row.get("expert_prompt_used")):
            v.append("M9_requires_expert_prompt")
        for s in ("Step1", "Step2", "Step3", "Step4", "Step5", "Step6"):
            if s not in steps:
                v.append(f"M9_missing_{s}")
        if s5 < 1:
            v.append("M9_requires_at_least_one_section")
        if fs < 1:
            v.append("M9_requires_final_sources")

    if retrieval:
        for k in ("step1_json_ok", "step2_json_ok", "step3_json_ok", "step4_json_ok"):
            if row.get(k) is False:
                v.append(f"{k}_false")
        if not bool(row.get("chapter_only_mode")) and row.get("step5_json_ok") is False and row.get("step5_selection_mode") != "rule_based":
            v.append("step5_json_ok_false")

    return (len(v) == 0), v


def load_batch_context(batch_dir: Path) -> BatchContext:
    batch_dir = Path(batch_dir)
    batch_id = batch_dir.name
    q_map: Dict[str, Dict[str, Any]] = {}
    pq = batch_dir / "parsed_questions.json"
    if pq.exists():
        arr = _load_json(pq)
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    qid = str(it.get("qid") or "").strip()
                    if qid:
                        q_map[qid] = it
    summary_map: Dict[str, Dict[str, Any]] = {}
    bs = batch_dir / "batch_summary.json"
    if bs.exists():
        d = _load_json(bs)
        for r in (d.get("results") or []):
            if isinstance(r, dict):
                qid = str(r.get("qid") or "").strip()
                if qid:
                    summary_map[qid] = r
    return BatchContext(batch_id=batch_id, batch_dir=batch_dir, q_map=q_map, summary_map=summary_map)


def compute_auto_metrics_for_question(ctx: BatchContext, qid: str) -> Dict[str, Any]:
    q_dir = ctx.batch_dir / qid
    qa_path = q_dir / "qa_result.json"
    payload: Dict[str, Any] = {}
    status = "ok"
    if (q_dir / "error.txt").exists() or not qa_path.exists():
        status = "failed"
    if qa_path.exists():
        try:
            payload = _load_json(qa_path)
        except Exception:
            status = "failed"
            payload = {}

    trace = _safe_list(payload.get("trace"))
    q_meta = _safe_dict(ctx.q_map.get(qid) or {})
    s_meta = _safe_dict(ctx.summary_map.get(qid) or {})

    answer1 = str(payload.get("answer1") or "")
    answer2 = str(payload.get("answer2") or "")
    selected = str(payload.get("selected") or "")
    answer1_issues = _integrity_issues(answer1)
    answer2_issues = _integrity_issues(answer2)
    selected_issues = answer1_issues if selected == "answer1" else answer2_issues

    chapter_summary_chars_total = 0
    for ch in _safe_list(payload.get("step4_selected_chapters")):
        if isinstance(ch, dict):
            chapter_summary_chars_total += len(str(ch.get("chapter_summary") or ""))
    evidence_chars_total = int(payload.get("section_chars_total") or 0)
    if evidence_chars_total <= 0 and bool(payload.get("chapter_only_mode")):
        evidence_chars_total = chapter_summary_chars_total

    row: Dict[str, Any] = {
        "schema_version": AUTO_METRICS_SCHEMA_VERSION,
        "batch_id": ctx.batch_id,
        "batch_dir": str(ctx.batch_dir.resolve()),
        "qid": qid,
        "idx": int(q_meta.get("idx") or 0),
        "title": str(q_meta.get("title") or ""),
        "status": status,
        "error_type": _classify_error(q_dir, qa_path if qa_path.exists() else None),
        "has_error_file": (q_dir / "error.txt").exists(),
        "has_report_error_file": (q_dir / "report_error.txt").exists(),
        "mode": str(payload.get("test_mode") or s_meta.get("mode") or ""),
        "executed_steps": _safe_list(payload.get("executed_steps")),
        "executed_steps_count": len(_safe_list(payload.get("executed_steps"))),
        "retrieval_used": bool(payload.get("retrieval_used")),
        "retrieval_rounds": int(payload.get("retrieval_rounds") or 0),
        "forced_retrieval": bool(payload.get("forced_retrieval")),
        "no_retrieval": bool(payload.get("no_retrieval")),
        "expert_prompt_used": bool(payload.get("expert_prompt_used")),
        "chapter_only_mode": bool(payload.get("chapter_only_mode")),
        "skip_section_ranking": bool(payload.get("skip_section_ranking")),
        "topk_books": int(payload.get("topk_books") or 0),
        "topk_books_view": int(payload.get("topk_books_view") or 0),
        "topk_candidate_chapters": int(payload.get("topk_candidate_chapters") or 0),
        "topk_selected_chapters": int(payload.get("topk_selected_chapters") or 0),
        "topk_sections": int(payload.get("topk_sections") or 0),
        "section_budget_used": int(payload.get("section_budget_used") or 0),
        "p_solve": payload.get("p_solve"),
        "confidence_threshold": payload.get("confidence_threshold"),
        "m5_gate_triggered_retrieval": payload.get("m5_gate_triggered_retrieval"),
        "step1_candidates_books_count": len(_safe_list(payload.get("step1_candidates_books"))),
        "step2_selected_books_count": len(_safe_list(payload.get("step2_selected_books"))),
        "step3_candidate_chapters_count": len(_safe_list(payload.get("step3_candidate_chapters"))),
        "step4_selected_chapters_count": len(_safe_list(payload.get("step4_selected_chapters"))),
        "step5_selected_sections_count": len(_safe_list(payload.get("step5_selected_sections"))),
        "step5_available_sections_total": int(payload.get("step5_available_sections_total") or 0),
        "final_sources_count": len(_safe_list(payload.get("final_sources"))),
        "section_chars_total": int(payload.get("section_chars_total") or 0),
        "evidence_chars_total": evidence_chars_total,
        "step5_selection_mode": str(payload.get("step5_selection_mode") or ""),
        "answer_raw_chars": len(str(payload.get("answer_raw") or "")),
        "final_answer_chars": len(str(payload.get("final_answer") or "")),
        "outline_chars": len(str(payload.get("outline") or "")),
        "audit_outline_chars": len(str(payload.get("audit_outline") or "")),
        "answer1_issues": answer1_issues,
        "answer2_issues": answer2_issues,
        "selected_issues": selected_issues,
        "selected_severely_broken": _is_severely_broken(selected_issues),
        "selection_override": _first_trace(trace, "SelectionOverride") is not None,
        "step1_json_ok": payload.get("step1_json_ok"),
        "step2_json_ok": payload.get("step2_json_ok"),
        "step3_json_ok": payload.get("step3_json_ok"),
        "step4_json_ok": payload.get("step4_json_ok"),
        "step5_json_ok": payload.get("step5_json_ok"),
        "duration_sec": payload.get("duration_sec") or s_meta.get("duration_sec") or "",
        "step_durations_sec": _safe_dict(payload.get("step_durations_sec")),
        "question_dir": str(q_dir.resolve()),
        "qa_result_path": str(qa_path.resolve()) if qa_path.exists() else "",
    }
    ok, violations = _check_spec(str(row.get("mode") or ""), row)
    row["spec_ok"] = ok
    row["spec_violations"] = violations
    return row


def iter_auto_metrics(ctx: BatchContext) -> Iterable[Dict[str, Any]]:
    qids = sorted(set(list(ctx.q_map.keys()) + [p.name for p in ctx.batch_dir.iterdir() if p.is_dir() and p.name.startswith("Q")]))
    for qid in qids:
        yield compute_auto_metrics_for_question(ctx, qid)


def _jsonish(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def run_auto_metrics(*, batch_dir: Path) -> Dict[str, Path]:
    ctx = load_batch_context(batch_dir)
    eval_dir = Path(batch_dir) / "_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_auto_metrics(ctx))
    jsonl_path = eval_dir / "auto_metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fieldnames = auto_metrics_fieldnames()
    csv_path = eval_dir / "auto_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _jsonish(r.get(k)) for k in fieldnames})
    return {"jsonl": jsonl_path, "csv": csv_path}
