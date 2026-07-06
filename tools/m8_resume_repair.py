#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pdf_txt_align.config import load_config
from pdf_txt_align.llm_calls import call_chat, call_json
from lookupbooks_sys.library import BookLibrary
from lookupbooks_sys.qa_agent import (
    MultiBookQAAgent,
    _answer_format_errors,
    _coerce_triplets,
    _integrity_issues,
    _is_severely_broken,
    _parse_letter,
    _truncate_chars,
    make_chapter_label,
    make_section_label,
)
from lookupbooks_sys.reporting import (
    build_item_from_qa_result,
    render_markdown_report,
    write_markdown_report,
)
from run import _build_pool


REPAIR_DIRNAME = "_repair"
RESUME_QIDS_TXT = "m8_resume_qids.txt"
FULL_RERUN_QIDS_TXT = "m8_full_rerun_qids.txt"
PLAN_CSV = "m8_repair_plan.csv"
REPORT_CSV = "m8_repair_report.csv"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _read_qid_file(path: Optional[Path]) -> List[str]:
    if not path or not path.exists():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        out.append(t)
    return out


def _write_qid_file(path: Path, qids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uniq = []
    seen = set()
    for qid in qids:
        q = str(qid).strip()
        if q and q not in seen:
            uniq.append(q)
            seen.add(q)
    path.write_text("\n".join(uniq) + ("\n" if uniq else ""), encoding="utf-8")


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v or "").strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _parse_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _parse_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _split_spec_violations(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v or "").strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    parts = [x.strip() for x in s.replace(";", ",").split(",")]
    return [x for x in parts if x]


def _build_question_map(batch_dir: Path) -> Dict[str, Dict[str, Any]]:
    parsed = _load_json(batch_dir / "parsed_questions.json", default=[])
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("qid") or "").strip()
            if qid:
                out[qid] = item
    return out


def _collect_q_dirs(batch_dir: Path) -> List[Path]:
    return sorted([p for p in batch_dir.iterdir() if p.is_dir() and p.name.upper().startswith("Q")], key=lambda x: x.name)


def _plan_rows(batch_dir: Path, *, include_failed_without_qa: bool = True) -> List[Dict[str, Any]]:
    q_map = _build_question_map(batch_dir)
    combined_rows = {str(r.get("qid") or "").strip(): r for r in _read_csv_rows(batch_dir / "_eval" / "combined.csv") if str(r.get("qid") or "").strip()}
    batch_summary = _load_json(batch_dir / "batch_summary.json", default={})
    summary_rows = {str(r.get("qid") or "").strip(): r for r in (batch_summary.get("results") or []) if isinstance(r, dict) and str(r.get("qid") or "").strip()}

    rows: List[Dict[str, Any]] = []
    for q_dir in _collect_q_dirs(batch_dir):
        qid = q_dir.name
        qa_path = q_dir / "qa_result.json"
        payload = _load_json(qa_path, default={}) if qa_path.exists() else {}
        combined = combined_rows.get(qid, {})
        summary = summary_rows.get(qid, {})
        test_mode = str(payload.get("test_mode") or combined.get("test_mode") or combined.get("mode") or summary.get("mode") or "").strip()
        step4 = payload.get("step4_selected_chapters") if isinstance(payload, dict) else []
        step5 = payload.get("step5_selected_sections") if isinstance(payload, dict) else []
        topk_sections = _parse_int(payload.get("topk_sections") if isinstance(payload, dict) else None, _parse_int(combined.get("topk_sections"), 0))
        avail_sections = _parse_int(payload.get("step5_available_sections_total") if isinstance(payload, dict) else None, _parse_int(combined.get("step5_available_sections_total"), 0))
        selected_sections = len(step5) if isinstance(step5, list) else _parse_int(combined.get("section_budget_used"), 0)
        spec_ok = _parse_bool(combined.get("spec_ok")) if combined else False
        status = str(combined.get("status") or summary.get("status") or ("ok" if qa_path.exists() else "missing"))
        violations = _split_spec_violations(combined.get("spec_violations") or payload.get("spec_violations"))
        has_budget_violation = "M8_section_budget_not_met" in violations
        has_qa = qa_path.exists() and isinstance(payload, dict) and bool(payload)
        has_step4 = isinstance(step4, list) and len(step4) > 0
        can_resume_repair = has_qa and has_step4 and test_mode.upper().startswith("M8")
        needs_resume_repair = bool(can_resume_repair and has_budget_violation)
        needs_full_rerun = False
        reason = ""
        if needs_resume_repair:
            reason = "M8_section_budget_not_met"
        elif include_failed_without_qa and (status != "ok" or not has_qa):
            if test_mode.upper().startswith("M8"):
                if can_resume_repair:
                    needs_resume_repair = True
                    reason = reason or f"status={status}"
                else:
                    needs_full_rerun = True
                    reason = f"{status}; no qa_result or no step4 cache"
        required_sections = min(max(0, topk_sections), max(0, avail_sections)) if test_mode.upper().startswith("M8") else 0
        rows.append({
            "qid": qid,
            "title": str((q_map.get(qid) or {}).get("title") or ""),
            "test_mode": test_mode,
            "status": status,
            "qa_exists": has_qa,
            "step4_exists": has_step4,
            "spec_ok": spec_ok,
            "spec_violations": json.dumps(violations, ensure_ascii=False),
            "topk_sections": topk_sections,
            "available_sections": avail_sections,
            "selected_sections": selected_sections,
            "required_sections": required_sections,
            "needs_resume_repair": needs_resume_repair,
            "needs_full_rerun": needs_full_rerun,
            "reason": reason,
        })
    return rows


def cmd_plan(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch_dir)
    repair_dir = _ensure_dir(batch_dir / REPAIR_DIRNAME)
    rows = _plan_rows(batch_dir, include_failed_without_qa=not bool(args.no_include_failed))
    fieldnames = [
        "qid", "title", "test_mode", "status", "qa_exists", "step4_exists", "spec_ok",
        "spec_violations", "topk_sections", "available_sections", "selected_sections",
        "required_sections", "needs_resume_repair", "needs_full_rerun", "reason",
    ]
    plan_csv = repair_dir / PLAN_CSV
    _write_csv(plan_csv, rows, fieldnames)

    resume_qids = [r["qid"] for r in rows if _parse_bool(r.get("needs_resume_repair"))]
    rerun_qids = [r["qid"] for r in rows if _parse_bool(r.get("needs_full_rerun"))]
    _write_qid_file(repair_dir / RESUME_QIDS_TXT, resume_qids)
    _write_qid_file(repair_dir / FULL_RERUN_QIDS_TXT, rerun_qids)

    print(f"[PLAN] plan_csv={plan_csv}")
    print(f"[PLAN] resume_repair_qids={len(resume_qids)} -> {repair_dir / RESUME_QIDS_TXT}")
    print(f"[PLAN] full_rerun_qids={len(rerun_qids)} -> {repair_dir / FULL_RERUN_QIDS_TXT}")


def _filtered_old_trace(old_trace: Any) -> List[Dict[str, Any]]:
    if not isinstance(old_trace, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in old_trace:
        if not isinstance(item, dict):
            continue
        step = str(item.get("step") or "")
        if step.startswith("Step5") or step.startswith("Step6") or step in {"Compare", "PostFormat", "SelectionOverride"}:
            continue
        out.append(item)
    return out


def _safe_copy_json(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _remove_if_exists(p: Path) -> None:
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def _load_q_payload(q_dir: Path) -> Dict[str, Any]:
    qa_path = q_dir / "qa_result.json"
    payload = _load_json(qa_path, default=None)
    if not isinstance(payload, dict):
        raise FileNotFoundError(f"qa_result.json missing or invalid: {qa_path}")
    return payload




def _resolve_book_id_mapping(agent: MultiBookQAAgent, payload: Dict[str, Any]) -> Dict[int, int]:
    """Map legacy per-question compact book ids (1/2/3) to actual library book ids.

    Old qa_result payloads may store per-question ranked book ids rather than the real
    ids from the library index. Resolve them by title against the current library.
    """
    mapping: Dict[int, int] = {}
    candidates = []
    for key in ("step2_selected_books", "step1_candidates_books"):
        rows = payload.get(key) or []
        if isinstance(rows, list):
            candidates.extend([r for r in rows if isinstance(r, dict)])
    for row in candidates:
        try:
            old_id = int(row.get("book_id"))
        except Exception:
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        ent = agent.lib.get_entry_by_title(title)
        if ent is not None:
            mapping[old_id] = int(ent.book_id)
    return mapping


def _apply_book_id_mapping_to_payload(payload: Dict[str, Any], mapping: Dict[int, int]) -> None:
    if not mapping:
        return
    for key in ("step1_candidates_books", "step2_selected_books", "step4_selected_chapters", "step5_selected_sections", "section_views"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                old_id = int(row.get("book_id"))
            except Exception:
                continue
            if old_id in mapping:
                row["book_id"] = mapping[old_id]

def _prepare_system_msgs(agent: MultiBookQAAgent, payload: Dict[str, Any]) -> List[Dict[str, str]]:
    expert_prompt_used = bool(payload.get("expert_prompt_used"))
    system_msgs: List[Dict[str, str]] = []
    if expert_prompt_used:
        exp = agent._load_expert_prompt().strip()
        if exp:
            system_msgs.append({"role": "system", "content": exp})
    return system_msgs


def _repair_one_qid(
    *,
    batch_dir: Path,
    qid: str,
    agent: MultiBookQAAgent,
    pool,
    dry_run: bool,
    purge_judge_cache: bool,
    refresh_reports: bool,
) -> Dict[str, Any]:
    q_dir = batch_dir / qid
    if not q_dir.exists():
        raise FileNotFoundError(f"question dir not found: {q_dir}")

    payload = _load_q_payload(q_dir)
    if str(payload.get("test_mode") or "").upper() != "M8":
        raise ValueError(f"{qid}: test_mode is not M8")
    book_id_mapping = _resolve_book_id_mapping(agent, payload)
    if book_id_mapping:
        _apply_book_id_mapping_to_payload(payload, book_id_mapping)
    step4_selected_chapters = payload.get("step4_selected_chapters") or []
    if not isinstance(step4_selected_chapters, list) or not step4_selected_chapters:
        raise ValueError(f"{qid}: no cached step4_selected_chapters, cannot resume from Step5")
    answer1 = str(payload.get("answer1") or "").strip()
    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError(f"{qid}: empty question")
    if not answer1:
        raise ValueError(f"{qid}: empty answer1")

    cfg = agent.cfg
    llm_default = str(getattr(getattr(cfg, "models"), "llm_model"))
    rank5_model = agent._model("rank_sections_stage5_model", agent._model("rank_chapters_model", llm_default))
    answer_model = agent._model("final_answer_model", llm_default)
    compare_model = agent._model("compare_model", llm_default)
    format_model = agent._model("format_model", llm_default)
    rank_max = int(agent._qa_val("rank_max_tokens", 1024) or 1024)
    final_max = int(agent._qa_val("final_answer_max_tokens", 8192) or 8192)
    compare_max = int(agent._qa_val("answer_compare_max_tokens", 64) or 64)
    post_format_max = int(agent._qa_val("post_format_max_tokens", 12000) or 12000)
    answer_format_repair_max = int(agent._qa_val("answer_format_repair_max_tokens", post_format_max) or post_format_max)
    allow_masked_sections = bool(agent._qa_val("allow_masked_sections", False))
    prefer_exposure = list(agent._qa_val("prefer_exposure", ["expose", "caution"]))
    min_section_quality = float(agent._qa_val("min_section_quality_score", 0.0) or 0.0)
    section_text_max_chars = int(agent._qa_val("section_text_max_chars", 80000) or 80000)
    sections_total_max_chars = int(agent._qa_val("sections_total_max_chars", 240000) or 240000)
    topk_sections = int(payload.get("topk_sections") or agent._qa_val("topk_sections", agent._qa_val("step5_sections_total_max", 10)) or 10)
    step5_sections_total_max = max(1, topk_sections)
    skip_section_ranking = bool(payload.get("skip_section_ranking"))
    forced_retrieval = bool(payload.get("forced_retrieval", True))

    system_msgs = _prepare_system_msgs(agent, payload)
    trace: List[Dict[str, Any]] = _filtered_old_trace(payload.get("trace"))
    repair_trace: List[Dict[str, Any]] = []
    repair_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    total_section_chars = 0
    step5_available_sections_total = 0
    step5_selected_sections: List[Dict[str, Any]] = []
    section_views: List[Dict[str, Any]] = []
    final_sources: List[Dict[str, Any]] = []
    legacy_extracts: List[str] = []

    def _log(step: str, **kw: Any) -> None:
        entry = {"step": step, **kw}
        repair_trace.append(entry)

    t0 = time.perf_counter()
    allowed_section_ids: Dict[Tuple[int, int], List[str]] = {}
    sec_lines: List[str] = []
    for ch_item in step4_selected_chapters:
        bid, chno = int(ch_item["book_id"]), int(ch_item["chapter_no"])
        secs = agent.lib.list_sections(
            bid,
            chno,
            filter_policy={
                "prefer_exposure": prefer_exposure,
                "allow_masked_sections": allow_masked_sections,
                "min_section_quality_score": min_section_quality,
            },
        )
        allowed_section_ids[(bid, chno)] = [s.section_id for s in secs]
        step5_available_sections_total += len(secs)
        for s in secs:
            sec_lines.append(
                f"book_id={bid}\nchapter_no={chno}\nsection_id={s.section_id}\n"
                f"section_label={make_section_label(ch_item['chapter_label'], s.title)}\n"
                f"summary={s.summary}\nkeywords={', '.join(s.keywords)}\n"
                f"exposure={s.exposure_decision}\nquality={s.quality_label}:{s.quality_score}"
            )

    sel_triplets: List[Tuple[int, int, str]] = []
    why5: Dict[str, str] = {}
    out5: Any = None
    step5_json_ok: Optional[bool] = None
    step5_selection_mode = ""
    required_sections = min(step5_sections_total_max, step5_available_sections_total)

    with pool.session() as session:
        if skip_section_ranking:
            for ch_item in step4_selected_chapters:
                pair = (int(ch_item["book_id"]), int(ch_item["chapter_no"]))
                for sid in allowed_section_ids.get(pair, []):
                    sel_triplets.append((pair[0], pair[1], sid))
                    why5[f"{pair[0]}:{pair[1]}:{sid}"] = "rule_based_fill"
                    if len(sel_triplets) >= required_sections:
                        break
                if len(sel_triplets) >= required_sections:
                    break
            step5_json_ok = True
            step5_selection_mode = "rule_based"
            out5 = {"selected": [[a, b, c] for a, b, c in sel_triplets], "why": why5}
        else:
            prompt5 = (
                f"根据下面的问题与候选节信息，选出最相关的节，总数不超过 {step5_sections_total_max}。\n"
                "输出 ONLY JSON: {\"selected\":[[book_id, chapter_no, section_id], ...], \"why\": {\"book_id:chapter_no:section_id\": \"理由\"}}。\n"
                "只能从给定候选节中选择；不得重复。\n\n"
                f"[问题]\n{question}\n\n[候选节]\n" + "\n\n".join(sec_lines)
            )
            out5 = call_json(session, rank5_model, system_msgs + [{"role": "user", "content": prompt5}], max_tokens=rank_max, temperature=0.0)
            step5_json_ok = isinstance(out5, dict) and isinstance(out5.get("selected"), list)
            sel_triplets = _coerce_triplets(out5.get("selected") if isinstance(out5, dict) else None, max_n=step5_sections_total_max, allowed_pairs=allowed_section_ids)
            why5 = out5.get("why") if isinstance(out5, dict) and isinstance(out5.get("why"), dict) else {}
            # Fixed backfill: only fill when the LLM selected fewer than the required budget.
            if len(sel_triplets) < required_sections:
                seen = set(sel_triplets)
                for ch_item in step4_selected_chapters:
                    pair = (int(ch_item["book_id"]), int(ch_item["chapter_no"]))
                    for sid in allowed_section_ids.get(pair, []):
                        item = (pair[0], pair[1], sid)
                        if item in seen:
                            continue
                        sel_triplets.append(item)
                        seen.add(item)
                        if len(sel_triplets) >= required_sections:
                            break
                    if len(sel_triplets) >= required_sections:
                        break
            step5_selection_mode = "llm_ranked"

        if forced_retrieval and len(sel_triplets) < required_sections:
            raise ValueError(
                f"{qid}: Section retrieval insufficient after repair: selected={len(sel_triplets)} required={required_sections} available={step5_available_sections_total}"
            )

        for bid, chno, sid in sel_triplets[:required_sections]:
            ch_item = next((x for x in step4_selected_chapters if int(x["book_id"]) == bid and int(x["chapter_no"]) == chno), None)
            secs = agent.lib.list_sections(
                bid,
                chno,
                filter_policy={
                    "prefer_exposure": prefer_exposure,
                    "allow_masked_sections": allow_masked_sections,
                    "min_section_quality_score": min_section_quality,
                },
            )
            sm = next((s for s in secs if s.section_id == sid), None)
            if not sm:
                continue
            entry = agent.lib.get_entry(bid)
            chapter_label = str(ch_item.get("chapter_label") or "") if isinstance(ch_item, dict) else ""
            if not chapter_label:
                chapter_label = make_chapter_label(entry.title if entry else f"book{bid}", chno, "")
            section_label = make_section_label(chapter_label, sm.title)
            text = _truncate_chars(agent.lib.read_section_text(bid, chno, sid), section_text_max_chars)
            if total_section_chars + len(text) > sections_total_max_chars:
                remain = max(0, sections_total_max_chars - total_section_chars)
                text = text[:remain]
            total_section_chars += len(text)
            view = {
                "book_id": bid,
                "chapter_no": chno,
                "section_id": sid,
                "section_title": sm.title,
                "section_label": section_label,
                "text_path": sm.file,
                "text_excerpt": text,
                "why": str(why5.get(f"{bid}:{chno}:{sid}") or ""),
            }
            step5_selected_sections.append({k: view[k] for k in ["book_id", "chapter_no", "section_id", "section_title", "section_label", "why"]})
            section_views.append(view)
            legacy_extracts.append(text)
            if total_section_chars >= sections_total_max_chars:
                break

        _log(
            "Step5.repair",
            repaired=True,
            step5_selection_mode=step5_selection_mode,
            topk_sections=topk_sections,
            available_sections_total=step5_available_sections_total,
            selected_sections_count=len(step5_selected_sections),
            required_sections=required_sections,
            prompt_selection=out5,
        )

        knowledge_blocks = []
        for v in section_views:
            knowledge_blocks.append(f"[来源] {v['section_label']}\n[内容]\n{v['text_excerpt']}\n")
            final_sources.append({"section_label": v["section_label"], "why_used": v.get("why", "")})
        prompt6 = (
            "根据下面按节组织的知识回答问题，并严格按如下格式输出：\n"
            "【解题思路与公式推导思路】\n- 说明将使用哪些来源、哪些关键公式与推导步骤。\n"
            "【最终答案】\n- 直接给出结论/表达式/数值；不同小题分段。\n\n"
            f"[问题]\n{question}\n\n[知识]\n" + "\n\n".join(knowledge_blocks)
        )
        answer2_raw = call_chat(session, answer_model, system_msgs + [{"role": "user", "content": prompt6}], max_tokens=final_max, temperature=0.2).strip()
        answer2 = agent._enforce_answer_format(
            session,
            stage_name="Step6.repair",
            question=question,
            raw_answer=answer2_raw,
            generation_model=answer_model,
            generation_prompt=prompt6,
            repair_model=format_model,
            system_msgs=system_msgs,
            generation_max_tokens=final_max,
            repair_max_tokens=answer_format_repair_max,
            trace=repair_trace,
        )

        cmp_prompt = (
            "比较下面同一问题的两个答案，优先级：正确性 > 完整性/无截断 > 表达清晰。"
            "若某答案有未闭合代码块、重复标题、明显截断尾巴，则判为更差。"
            "只返回 A 或 B。A=答案1更好，B=答案2更好。\n\n"
            f"[问题]\n{question}\n\n[答案1]\n{answer1}\n\n[答案2]\n{answer2}\n"
        )
        cmp_raw = call_chat(session, compare_model, system_msgs + [{"role": "user", "content": cmp_prompt}], max_tokens=compare_max, temperature=0.0)
        cmp = _parse_letter(cmp_raw, "AB")
        selected = "answer1" if cmp == "A" else "answer2"
        issues1 = _integrity_issues(answer1)
        issues2 = _integrity_issues(answer2)
        if selected == "answer1" and _is_severely_broken(issues1) and not _is_severely_broken(issues2):
            selected = "answer2"
            _log("SelectionOverride", from_selected="answer1", to_selected="answer2", reason=issues1)
        if selected == "answer2" and _is_severely_broken(issues2) and not _is_severely_broken(issues1):
            selected = "answer1"
            _log("SelectionOverride", from_selected="answer2", to_selected="answer1", reason=issues2)
        _log("Step6.compare.repair", raw=cmp_raw, choice=cmp, selected=selected, answer1_issues=issues1, answer2_issues=issues2)

        final_raw = answer1 if selected == "answer1" else answer2
        retrieval_context = {
            "step1_candidates_books": payload.get("step1_candidates_books") or [],
            "step2_selected_books": payload.get("step2_selected_books") or [],
            "step3_candidate_chapters": payload.get("step3_candidate_chapters") or [],
            "step4_selected_chapters": step4_selected_chapters,
            "step5_selected_sections": step5_selected_sections,
            "section_views": section_views,
            "final_sources": final_sources,
        }
        fmt = agent._post_format_answer(session, question, final_raw, retrieval_context=retrieval_context, model=format_model, max_tokens=post_format_max)
        final = f"【解题思路与公式推导思路】\n{fmt['final_reasoning']}\n\n【最终答案】\n{fmt['final_answer']}".strip()

    repair_duration_sec = round(time.perf_counter() - t0, 6)
    new_payload = dict(payload)
    new_payload["answer"] = final
    new_payload["answer2"] = answer2
    new_payload["selected"] = selected
    new_payload["outline"] = fmt["final_reasoning"]
    new_payload["audit_outline"] = fmt["audit_reasoning"]
    new_payload["final_answer"] = fmt["final_answer"]
    new_payload["answer_raw"] = final_raw
    new_payload["step5_selected_sections"] = step5_selected_sections
    new_payload["section_views"] = section_views
    new_payload["final_sources"] = final_sources
    new_payload["extracts"] = legacy_extracts
    new_payload["retrieval_rounds"] = len([x for x in section_views if str(x.get("text_excerpt") or "").strip()])
    new_payload["section_budget_used"] = len(step5_selected_sections)
    new_payload["step5_selection_mode"] = step5_selection_mode
    new_payload["section_chars_total"] = total_section_chars
    new_payload["step5_available_sections_total"] = step5_available_sections_total
    new_payload["step5_json_ok"] = step5_json_ok
    new_payload["repair_duration_sec"] = repair_duration_sec
    new_payload["repair_started_at"] = repair_started_at
    new_payload["repair_finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    new_payload["repair_mode"] = "resume_from_existing_qa_step5_step6"
    new_payload["duration_sec_original_full_run"] = payload.get("duration_sec")
    new_payload["step_durations_sec_original_full_run"] = payload.get("step_durations_sec")
    new_payload["trace_before_m8_repair_preserved"] = True
    new_payload["trace"] = trace + repair_trace

    qa_path = q_dir / "qa_result.json"
    backup_path = q_dir / "qa_result.before_m8_resume_repair.json"
    if not backup_path.exists() and not dry_run:
        _safe_copy_json(qa_path, backup_path)

    if not dry_run:
        _write_json(qa_path, new_payload)
        if purge_judge_cache:
            _remove_if_exists(batch_dir / "_eval" / "judge_outputs" / f"{qid}.json")
            _remove_if_exists(batch_dir / "_eval" / "judge_inputs" / f"{qid}.json")
        if refresh_reports:
            q_map = _build_question_map(batch_dir)
            q_item = q_map.get(qid) or {}
            title = str(q_item.get("title") or qid)
            item_r = build_item_from_qa_result(qid=qid, title=title, question=question, qa_result_payload=new_payload)
            meta = {
                "batch_id": batch_dir.name,
                "batch_dir": str(batch_dir.resolve()),
                "repair_mode": "resume_from_existing_qa_step5_step6",
            }
            md1 = render_markdown_report([item_r], title=f"lookupbooks_sys 单题修复报告（{qid}）", meta=meta)
            md2 = render_markdown_report([item_r], title=f"lookupbooks_sys 单题修复报告（审查版｜{qid}）", meta=meta, mode="audit")
            write_markdown_report(q_dir / "report.md", md1)
            write_markdown_report(q_dir / "report_audit.md", md2)

    return {
        "qid": qid,
        "status": "ok" if not dry_run else "dry_run_ok",
        "topk_sections": topk_sections,
        "available_sections_total": step5_available_sections_total,
        "selected_sections_count": len(step5_selected_sections),
        "required_sections": required_sections,
        "selected": selected,
        "repair_duration_sec": repair_duration_sec,
        "backup_path": str(backup_path),
        "qa_path": str(qa_path),
    }


def _resolve_qids(batch_dir: Path, args: argparse.Namespace) -> List[str]:
    qids: List[str] = []
    for q in (args.qid or []):
        qids.append(str(q).strip())
    qid_file = Path(args.qid_file) if getattr(args, "qid_file", None) else None
    qids.extend(_read_qid_file(qid_file))
    if not qids:
        default_file = batch_dir / REPAIR_DIRNAME / RESUME_QIDS_TXT
        qids.extend(_read_qid_file(default_file))
    uniq: List[str] = []
    seen = set()
    for q in qids:
        if q and q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq




def _already_repaired_ok(q_dir: Path) -> bool:
    payload = _load_json(q_dir / "qa_result.json", default={})
    if not isinstance(payload, dict):
        return False
    if str(payload.get("repair_mode") or "") != "resume_from_existing_qa_step5_step6":
        return False
    try:
        topk = int(payload.get("topk_sections") or 0)
        avail = int(payload.get("step5_available_sections_total") or 0)
    except Exception:
        return False
    selected = payload.get("step5_selected_sections") or []
    if not isinstance(selected, list):
        return False
    required = min(max(0, topk), max(0, avail))
    return len(selected) == required


def _filter_skippable_repaired(batch_dir: Path, qids: List[str], *, skip_if_repaired: bool) -> Tuple[List[str], List[str]]:
    if not skip_if_repaired:
        return qids, []
    todo: List[str] = []
    skipped: List[str] = []
    for qid in qids:
        q_dir = batch_dir / qid
        if _already_repaired_ok(q_dir):
            skipped.append(qid)
        else:
            todo.append(qid)
    return todo, skipped


def cmd_repair(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch_dir)
    qids = _resolve_qids(batch_dir, args)
    if not qids:
        raise SystemExit("No qids to repair. Run 'plan' first or pass --qid / --qid-file.")

    workers = max(1, int(getattr(args, "workers", 1) or 1))
    qids, skipped_qids = _filter_skippable_repaired(
        batch_dir,
        qids,
        skip_if_repaired=not bool(getattr(args, "no_skip_if_repaired", False)),
    )
    if not qids:
        print(f"[REPAIR] nothing to do; skipped_already_repaired={len(skipped_qids)}")
        return

    cfg = load_config(args.agent_config)
    pool = _build_pool(cfg)
    repair_dir = _ensure_dir(batch_dir / REPAIR_DIRNAME)
    tls = local()

    def _get_agent() -> MultiBookQAAgent:
        agent = getattr(tls, "agent", None)
        if agent is None:
            lib = BookLibrary(Path(args.library_root))
            agent = MultiBookQAAgent(lib, cfg)
            tls.agent = agent
        return agent

    total = len(qids)
    report_rows: List[Dict[str, Any]] = []
    for qid in skipped_qids:
        report_rows.append({"qid": qid, "status": "skipped_already_repaired"})

    def _job(ix_qid: Tuple[int, str]) -> Dict[str, Any]:
        idx, qid = ix_qid
        print(f"[REPAIR] START {idx}/{total} {qid}", flush=True)
        try:
            row = _repair_one_qid(
                batch_dir=batch_dir,
                qid=qid,
                agent=_get_agent(),
                pool=pool,
                dry_run=bool(args.dry_run),
                purge_judge_cache=not bool(args.no_purge_judge_cache),
                refresh_reports=not bool(args.no_refresh_reports),
            )
            print(f"[REPAIR] DONE  {idx}/{total} {qid}", flush=True)
            return row
        except Exception as e:
            err_path = repair_dir / f"{qid}.error.txt"
            err_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[REPAIR] FAILED {idx}/{total} {qid}: {e}", flush=True)
            return {
                "qid": qid,
                "status": "failed",
                "error": str(e),
                "traceback": str(err_path),
            }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, (idx, qid)) for idx, qid in enumerate(qids, start=1)]
        for fut in as_completed(futs):
            report_rows.append(fut.result())

    fieldnames = sorted({k for row in report_rows for k in row.keys()})
    report_csv = repair_dir / REPORT_CSV
    _write_csv(report_csv, sorted(report_rows, key=lambda r: str(r.get("qid") or "")), fieldnames)
    ok_cnt = sum(1 for r in report_rows if str(r.get("status")) in {"ok", "dry_run_ok"})
    fail_cnt = sum(1 for r in report_rows if str(r.get("status")) == "failed")
    skip_cnt = sum(1 for r in report_rows if str(r.get("status")) == "skipped_already_repaired")
    print(f"[REPAIR] workers={workers}")
    print(f"[REPAIR] report_csv={report_csv}")
    print(f"[REPAIR] ok={ok_cnt} fail={fail_cnt} skipped={skip_cnt}")


def cmd_export_rerun_batch(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch_dir)
    q_map = _build_question_map(batch_dir)
    qid_file = Path(args.qid_file) if args.qid_file else (batch_dir / REPAIR_DIRNAME / FULL_RERUN_QIDS_TXT)
    qids = _read_qid_file(qid_file)
    if not qids:
        raise SystemExit(f"No qids found in {qid_file}")
    out_md = Path(args.out_md) if args.out_md else (batch_dir / REPAIR_DIRNAME / "m8_full_rerun_q0.md")
    lines: List[str] = []
    missing: List[str] = []
    for qid in qids:
        item = q_map.get(qid)
        if not item:
            missing.append(qid)
            continue
        title = str(item.get("title") or qid)
        question = str(item.get("question") or "").rstrip()
        lines.append(f"## {qid}｜{title}")
        lines.append("")
        lines.append("```q0")
        lines.append(question)
        lines.append("```")
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[EXPORT] out_md={out_md}")
    if missing:
        print(f"[EXPORT] missing_in_parsed_questions={missing}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="M8 节预算修复：从旧 qa_result.json 续跑 Step5/Step6，并导出无法续跑题目的子批次。"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("plan", help="扫描 batch_dir，生成可续跑修复 / 需全量子批次重跑的 qid 清单")
    sp.add_argument("--batch-dir", required=True, help="Original M8 batch directory")
    sp.add_argument("--no-include-failed", action="store_true", help="Do not mark failed/no-qa M8 items as full rerun candidates")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("repair", help="从旧 qa_result 续跑 Step5/Step6，覆盖写回 qa_result.json")
    sp.add_argument("--batch-dir", required=True, help="Original M8 batch directory")
    sp.add_argument("--agent-config", default="./config/agent.yaml", help="Agent YAML config")
    sp.add_argument("--library-root", default="./library", help="Library root")
    sp.add_argument("--qid", action="append", default=[], help="Specific qid to repair (repeatable)")
    sp.add_argument("--qid-file", default=None, help="Text file listing qids to repair")
    sp.add_argument("--dry-run", action="store_true", help="Only simulate; do not overwrite qa_result or delete judge cache")
    sp.add_argument("--workers", type=int, default=1, help="Concurrent repair workers; each worker uses one API key at a time")
    sp.add_argument("--no-skip-if-repaired", action="store_true", help="Do not skip qids already repaired successfully")
    sp.add_argument("--no-purge-judge-cache", action="store_true", help="Do not delete _eval/judge_outputs/Qxxx.json for repaired qids")
    sp.add_argument("--no-refresh-reports", action="store_true", help="Do not rewrite report.md / report_audit.md")
    sp.set_defaults(func=cmd_repair)

    sp = sub.add_parser("export-rerun-batch", help="把无法续跑修复的 qid 导出为新的 q0 markdown 子批次")
    sp.add_argument("--batch-dir", required=True, help="Original M8 batch directory")
    sp.add_argument("--qid-file", default=None, help="Text file listing qids to rerun; default=_repair/m8_full_rerun_qids.txt")
    sp.add_argument("--out-md", default=None, help="Output markdown path")
    sp.set_defaults(func=cmd_export_rerun_batch)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
