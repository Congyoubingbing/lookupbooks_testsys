# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List

AUTO_METRICS_SCHEMA_VERSION = "auto_metrics_v2_section"

AUTO_METRICS_FIELDS: List[Dict[str, Any]] = [
    {"name": "schema_version", "type": "str", "desc": "schema version"},
    {"name": "batch_id", "type": "str", "desc": "batch id"},
    {"name": "batch_dir", "type": "str", "desc": "batch dir"},
    {"name": "qid", "type": "str", "desc": "question id"},
    {"name": "idx", "type": "int", "desc": "question index"},
    {"name": "title", "type": "str", "desc": "question title"},
    {"name": "status", "type": "str", "desc": "ok|failed"},
    {"name": "error_type", "type": "str", "desc": "failure classification"},
    {"name": "has_error_file", "type": "bool", "desc": "error.txt exists"},
    {"name": "has_report_error_file", "type": "bool", "desc": "report_error.txt exists"},
    {"name": "mode", "type": "str", "desc": "test mode"},
    {"name": "executed_steps", "type": "list[str]", "desc": "executed steps"},
    {"name": "executed_steps_count", "type": "int", "desc": "count"},
    {"name": "retrieval_used", "type": "bool", "desc": "retrieval branch used"},
    {"name": "retrieval_rounds", "type": "int", "desc": "non-empty selected sections"},
    {"name": "forced_retrieval", "type": "bool", "desc": "forced retrieval flag"},
    {"name": "no_retrieval", "type": "bool", "desc": "no retrieval flag"},
    {"name": "expert_prompt_used", "type": "bool", "desc": "expert prompt injected"},
    {"name": "chapter_only_mode", "type": "bool", "desc": "chapter-only ablation"},
    {"name": "skip_section_ranking", "type": "bool", "desc": "rule-based section selection"},
    {"name": "topk_books", "type": "int", "desc": "step1 max books"},
    {"name": "topk_books_view", "type": "int", "desc": "step2 books to view"},
    {"name": "topk_candidate_chapters", "type": "int", "desc": "step3 per-book chapter candidates"},
    {"name": "topk_selected_chapters", "type": "int", "desc": "step4 final chapters"},
    {"name": "topk_sections", "type": "int", "desc": "step5 section budget"},
    {"name": "section_budget_used", "type": "int", "desc": "selected section count"},
    {"name": "p_solve", "type": "float", "desc": "M5 p_solve"},
    {"name": "confidence_threshold", "type": "float", "desc": "M5 threshold"},
    {"name": "m5_gate_triggered_retrieval", "type": "bool", "desc": "M5 gate branch"},
    {"name": "step1_candidates_books_count", "type": "int", "desc": "count"},
    {"name": "step2_selected_books_count", "type": "int", "desc": "count"},
    {"name": "step3_candidate_chapters_count", "type": "int", "desc": "count"},
    {"name": "step4_selected_chapters_count", "type": "int", "desc": "count"},
    {"name": "step5_selected_sections_count", "type": "int", "desc": "count"},
    {"name": "step5_available_sections_total", "type": "int", "desc": "candidate section pool size"},
    {"name": "final_sources_count", "type": "int", "desc": "final sources count"},
    {"name": "section_chars_total", "type": "int", "desc": "total chars of selected section evidence"},
    {"name": "evidence_chars_total", "type": "int", "desc": "legacy alias to section_chars_total"},
    {"name": "step5_selection_mode", "type": "str", "desc": "llm_ranked|rule_based|chapter_only|none"},
    {"name": "answer_raw_chars", "type": "int", "desc": "chars"},
    {"name": "final_answer_chars", "type": "int", "desc": "chars"},
    {"name": "outline_chars", "type": "int", "desc": "chars"},
    {"name": "audit_outline_chars", "type": "int", "desc": "chars"},
    {"name": "answer1_issues", "type": "list[str]", "desc": "integrity issues"},
    {"name": "answer2_issues", "type": "list[str]", "desc": "integrity issues"},
    {"name": "selected_issues", "type": "list[str]", "desc": "selected issues"},
    {"name": "selected_severely_broken", "type": "bool", "desc": "selected broken"},
    {"name": "selection_override", "type": "bool", "desc": "selection override used"},
    {"name": "step1_json_ok", "type": "bool", "desc": "step1 json ok"},
    {"name": "step2_json_ok", "type": "bool", "desc": "step2 json ok"},
    {"name": "step3_json_ok", "type": "bool", "desc": "step3 json ok"},
    {"name": "step4_json_ok", "type": "bool", "desc": "step4 json ok"},
    {"name": "step5_json_ok", "type": "bool", "desc": "step5 json ok"},
    {"name": "duration_sec", "type": "float", "desc": "wall time"},
    {"name": "step_durations_sec", "type": "dict", "desc": "per-step durations"},
    {"name": "spec_ok", "type": "bool", "desc": "mode spec pass"},
    {"name": "spec_violations", "type": "list[str]", "desc": "violations"},
    {"name": "question_dir", "type": "str", "desc": "question dir"},
    {"name": "qa_result_path", "type": "str", "desc": "qa result path"},
]

JUDGE_SCHEMA_VERSION = "judge_v1"
JUDGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "schema_version": JUDGE_SCHEMA_VERSION,
    "qid": "Q011",
    "mode": "M3",
    "scores": {
        "overall": 0,
        "correctness": 0,
        "completeness": 0,
        "derivation": 0,
        "clarity": 0,
        "grounding": 0,
        "hallucination_resistance": 0,
    },
    "should_use_tools": True,
    "confidence": 0.75,
    "used_evidence_indices": [1, 2],
    "unsupported_claims": ["..."],
    "key_issues": ["..."],
    "strengths": ["..."],
}


def auto_metrics_fieldnames() -> List[str]:
    return [f["name"] for f in AUTO_METRICS_FIELDS]
