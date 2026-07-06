# -*- coding: utf-8 -*-
"""LLM-as-a-judge scoring for batch runs.

This module implements proxy evaluation (NOT ground truth).
It reads per-question qa_result.json and calls an LLM with a rubric prompt.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pdf_txt_align.llm_calls import call_chat

from .schemas import JUDGE_SCHEMA_VERSION


def _sha1_text(s: str) -> str:
    return sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _extract_json_obj(raw: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errs: List[str] = []
    if not raw or not isinstance(raw, str):
        return None, ["empty_raw"]
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None, ["no_json_braces"]
    cand = raw[start:end + 1]
    try:
        obj = json.loads(cand)
        return (obj, []) if isinstance(obj, dict) else (None, ["json_not_object"])
    except Exception:
        return None, ["json_parse_error"]


def _validate_judge_obj(obj: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    if not isinstance(obj, dict):
        return ["not_dict"]
    if obj.get("schema_version") != JUDGE_SCHEMA_VERSION:
        errs.append("schema_version_mismatch")
    for k in ["qid", "mode", "scores", "should_use_tools", "confidence", "used_evidence_indices", "key_issues", "strengths"]:
        if k not in obj:
            errs.append(f"missing_{k}")
    scores = obj.get("scores")
    if not isinstance(scores, dict):
        errs.append("scores_not_dict")
    else:
        for k in ["overall", "correctness", "completeness", "derivation", "clarity", "grounding", "hallucination_resistance"]:
            if k not in scores:
                errs.append(f"scores_missing_{k}")
            else:
                try:
                    iv = int(scores.get(k))
                    if iv < 0 or iv > 10:
                        errs.append(f"scores_out_of_range_{k}")
                except Exception:
                    errs.append(f"scores_not_int_{k}")
    try:
        cf = float(obj.get("confidence"))
        if cf < 0.0 or cf > 1.0:
            errs.append("confidence_out_of_range")
    except Exception:
        errs.append("confidence_not_float")
    if not isinstance(obj.get("should_use_tools"), bool):
        errs.append("should_use_tools_not_bool")
    uei = obj.get("used_evidence_indices")
    if not isinstance(uei, list):
        errs.append("used_evidence_indices_not_list")
    else:
        for x in uei:
            try:
                int(x)
            except Exception:
                errs.append("used_evidence_indices_not_int")
                break
    for lk in ["key_issues", "strengths", "unsupported_claims"]:
        if lk in obj and obj.get(lk) is not None and not isinstance(obj.get(lk), list):
            errs.append(f"{lk}_not_list")
    return errs


def _truncate(s: str, n: int) -> str:
    s = str(s or "")
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + "\n...(truncated)..."


def _build_evidence(qa_payload: Dict[str, Any], *, per_chunk_max: int, total_max: int) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    total = 0

    section_views = qa_payload.get("section_views")
    if isinstance(section_views, list) and section_views:
        for i, v in enumerate(section_views, start=1):
            if not isinstance(v, dict):
                continue
            t = _truncate(str(v.get("text_excerpt") or ""), per_chunk_max)
            if not t.strip():
                continue
            if total_max > 0 and total + len(t) > total_max:
                remain = max(0, total_max - total)
                t = _truncate(t, remain)
            total += len(t)
            evidence.append({
                "index": i,
                "kind": "section",
                "source": str(v.get("section_label") or v.get("source_label") or ""),
                "text": t,
            })
            if total_max > 0 and total >= total_max:
                break
        return evidence

    step4 = qa_payload.get("step4_selected_chapters")
    if isinstance(step4, list) and step4:
        for i, ch in enumerate(step4, start=1):
            if not isinstance(ch, dict):
                continue
            t = str(ch.get("chapter_title") or "")
            why = str(ch.get("why") or "")
            if why:
                t += "\n理由：" + why
            t = _truncate(t, per_chunk_max)
            if not t.strip():
                continue
            if total_max > 0 and total + len(t) > total_max:
                remain = max(0, total_max - total)
                t = _truncate(t, remain)
            total += len(t)
            evidence.append({
                "index": i,
                "kind": "chapter_summary",
                "source": str(ch.get("chapter_label") or ch.get("source_label") or ""),
                "text": t,
            })
            if total_max > 0 and total >= total_max:
                break
        if evidence:
            return evidence

    extracts = qa_payload.get("extracts")
    if not isinstance(extracts, list):
        extracts = []
    for i, ex in enumerate(extracts, start=1):
        t = _truncate(str(ex or ""), per_chunk_max)
        if not t.strip():
            continue
        if total_max > 0 and total + len(t) > total_max:
            remain = max(0, total_max - total)
            t = _truncate(t, remain)
        total += len(t)
        evidence.append({"index": i, "kind": "legacy", "source": "", "text": t})
        if total_max > 0 and total >= total_max:
            break
    return evidence


def build_judge_input(qid: str, qa_payload: Dict[str, Any], *, limits: Dict[str, int]) -> Dict[str, Any]:
    mode = str(qa_payload.get("test_mode") or "")
    final_answer = _truncate(str(qa_payload.get("final_answer") or qa_payload.get("answer") or ""), int(limits.get("final_answer_max_chars", 6000)))
    outline = _truncate(str(qa_payload.get("outline") or ""), int(limits.get("outline_max_chars", 8000)))
    audit_outline = _truncate(str(qa_payload.get("audit_outline") or ""), int(limits.get("audit_outline_max_chars", 12000)))
    return {
        "qid": qid,
        "mode": mode,
        "question": str(qa_payload.get("question") or ""),
        "final_answer": final_answer,
        "outline": outline,
        "audit_outline": audit_outline,
        "selected": str(qa_payload.get("selected") or ""),
        "tool_trace": {
            "retrieval_used": bool(qa_payload.get("retrieval_used")),
            "step1_candidates_books": qa_payload.get("step1_candidates_books") or [],
            "step2_selected_books": qa_payload.get("step2_selected_books") or [],
            "step3_candidate_chapters": qa_payload.get("step3_candidate_chapters") or [],
            "step4_selected_chapters": qa_payload.get("step4_selected_chapters") or [],
            "step5_selected_sections": qa_payload.get("step5_selected_sections") or [],
            "retrieval_rounds": int(qa_payload.get("retrieval_rounds") or 0),
            "forced_retrieval": bool(qa_payload.get("forced_retrieval")),
            "no_retrieval": bool(qa_payload.get("no_retrieval")),
            "chapter_only_mode": bool(qa_payload.get("chapter_only_mode")),
            "skip_section_ranking": bool(qa_payload.get("skip_section_ranking")),
            "topk_sections": qa_payload.get("topk_sections"),
            "p_solve": qa_payload.get("p_solve"),
        },
        "evidence": _build_evidence(
            qa_payload,
            per_chunk_max=int(limits.get("evidence_max_chars_per_chunk", 8000)),
            total_max=int(limits.get("evidence_max_chars_total", 24000)),
        ),
    }


def _flatten_scores(parsed: Dict[str, Any]) -> Dict[str, Any]:
    scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}
    return {
        "schema_version": parsed.get("schema_version"),
        "qid": parsed.get("qid"),
        "mode": parsed.get("mode"),
        "judge_score_overall": scores.get("overall"),
        "judge_score_correctness": scores.get("correctness"),
        "judge_score_completeness": scores.get("completeness"),
        "judge_score_derivation": scores.get("derivation"),
        "judge_score_clarity": scores.get("clarity"),
        "judge_score_grounding": scores.get("grounding"),
        "judge_score_hallucination_resistance": scores.get("hallucination_resistance"),
        "judge_should_use_tools": parsed.get("should_use_tools"),
        "judge_confidence": parsed.get("confidence"),
        "judge_used_evidence_indices": parsed.get("used_evidence_indices"),
        "judge_unsupported_claims": parsed.get("unsupported_claims"),
        "judge_key_issues": parsed.get("key_issues"),
        "judge_strengths": parsed.get("strengths"),
    }


def run_batch_judge(*, batch_dir: Path, pool, cfg, rubric_path: Path, skip_existing: bool = True, workers: int = 1) -> Dict[str, Path]:
    batch_dir = Path(batch_dir)
    eval_dir = _ensure_dir(batch_dir / "_eval")
    inputs_dir = _ensure_dir(eval_dir / "judge_inputs")
    outputs_dir = _ensure_dir(eval_dir / "judge_outputs")

    rubric_text = Path(rubric_path).read_text(encoding="utf-8")
    rubric_sha = _sha1_text(rubric_text)
    limits = {
        "final_answer_max_chars": int(getattr(cfg.eval, "final_answer_max_chars", 6000) or 6000),
        "outline_max_chars": int(getattr(cfg.eval, "outline_max_chars", 8000) or 8000),
        "audit_outline_max_chars": int(getattr(cfg.eval, "audit_outline_max_chars", 12000) or 12000),
        "evidence_max_chars_per_chunk": int(getattr(cfg.eval, "evidence_max_chars_per_chunk", 8000) or 8000),
        "evidence_max_chars_total": int(getattr(cfg.eval, "evidence_max_chars_total", 24000) or 24000),
    }
    judge_model = str(getattr(cfg.eval, "judge_model", getattr(cfg.models, "llm_model", "")) or getattr(cfg.models, "llm_model", ""))
    judge_temp = float(getattr(cfg.eval, "judge_temperature", 0.0) or 0.0)
    judge_max_tokens = int(getattr(cfg.eval, "judge_max_tokens", 900) or 900)

    q_dirs = sorted([p for p in batch_dir.iterdir() if p.is_dir() and p.name.upper().startswith("Q")])
    rows_map: Dict[str, Dict[str, Any]] = {}
    tasks: List[Tuple[str, Dict[str, Any], Dict[str, Any], Path]] = []

    for q_dir in q_dirs:
        qid = q_dir.name
        qa_path = q_dir / "qa_result.json"
        if not qa_path.exists():
            continue
        qa_payload = _load_json(qa_path)
        judge_input = build_judge_input(qid, qa_payload, limits=limits)
        (inputs_dir / f"{qid}.json").write_text(json.dumps(judge_input, ensure_ascii=False, indent=2), encoding="utf-8")
        out_path = outputs_dir / f"{qid}.json"
        cached = _load_json(out_path) if skip_existing and out_path.exists() else None
        if isinstance(cached, dict) and cached.get("rubric_sha") == rubric_sha and isinstance(cached.get("parsed"), dict):
            rows_map[qid] = _flatten_scores(cached["parsed"])
        else:
            tasks.append((qid, qa_payload, judge_input, out_path))

    def _judge_one(task: Tuple[str, Dict[str, Any], Dict[str, Any], Path]) -> Tuple[str, Dict[str, Any]]:
        qid, qa_payload, judge_input, out_path = task
        msgs = [
            {"role": "system", "content": rubric_text},
            {"role": "user", "content": "请严格按要求返回 ONLY JSON。\n\n" + json.dumps(judge_input, ensure_ascii=False, indent=2)},
        ]
        errs: List[str] = []
        try:
            with pool.session() as session:
                raw = call_chat(session, judge_model, msgs, max_tokens=judge_max_tokens, temperature=judge_temp)
            parsed, errs = _extract_json_obj(raw)
            errs = errs + (_validate_judge_obj(parsed) if isinstance(parsed, dict) else [])
            if errs:
                with pool.session() as session:
                    raw = call_chat(
                        session,
                        judge_model,
                        msgs + [{"role": "user", "content": "上一次输出未通过校验。请只返回满足 schema 的 JSON 对象，不要附加解释。"}],
                        max_tokens=judge_max_tokens,
                        temperature=0.0,
                    )
                parsed, errs = _extract_json_obj(raw)
                errs = errs + (_validate_judge_obj(parsed) if isinstance(parsed, dict) else [])
            if not isinstance(parsed, dict):
                parsed = {
                    "schema_version": JUDGE_SCHEMA_VERSION,
                    "qid": qid,
                    "mode": str(qa_payload.get("test_mode") or ""),
                    "scores": {k: 0 for k in ["overall", "correctness", "completeness", "derivation", "clarity", "grounding", "hallucination_resistance"]},
                    "should_use_tools": False,
                    "confidence": 0.0,
                    "used_evidence_indices": [],
                    "unsupported_claims": ["judge_parse_failed"],
                    "key_issues": ["judge_parse_failed"],
                    "strengths": [],
                }
                errs = ["judge_parse_failed"]
        except Exception as exc:
            parsed = {
                "schema_version": JUDGE_SCHEMA_VERSION,
                "qid": qid,
                "mode": str(qa_payload.get("test_mode") or ""),
                "scores": {k: 0 for k in ["overall", "correctness", "completeness", "derivation", "clarity", "grounding", "hallucination_resistance"]},
                "should_use_tools": False,
                "confidence": 0.0,
                "used_evidence_indices": [],
                "unsupported_claims": ["judge_request_failed"],
                "key_issues": [str(exc)],
                "strengths": [],
            }
            errs = ["judge_request_failed", str(exc)]
        out_payload = {
            "qid": qid,
            "rubric_path": str(rubric_path),
            "rubric_sha": rubric_sha,
            "judge_input": judge_input,
            "parsed": parsed,
            "validation_errors": errs,
        }
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return qid, _flatten_scores(parsed)

    max_workers = max(1, min(int(workers or 1), len(getattr(pool, "keys", []) or [1]), len(tasks) if tasks else 1))
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_judge_one, t) for t in tasks]
            for fut in as_completed(futs):
                qid, row = fut.result()
                rows_map[qid] = row

    ordered_qids = [p.name for p in q_dirs if p.name in rows_map]
    rows: List[Dict[str, Any]] = [rows_map[qid] for qid in ordered_qids]

    jsonl_path = eval_dir / "judge_scores.jsonl"
    csv_path = eval_dir / "judge_scores.csv"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return {"jsonl": jsonl_path, "csv": csv_path}
