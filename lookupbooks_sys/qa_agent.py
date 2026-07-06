from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pdf_txt_align.llm_calls import call_chat, call_json

from .library import BookLibrary


def _now_ts() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_letter(s: str, letters: str) -> str:
    s = (s or "").strip().upper()
    m = re.search(rf"\b([{re.escape(letters)}])\b", s)
    if m:
        return m.group(1)
    return letters[-1]


def _coerce_str_list(v: Any) -> List[str]:
    if not isinstance(v, list):
        return []
    out: List[str] = []
    for x in v:
        s = str(x or "").strip()
        if s:
            out.append(s)
    return out

def _indent_block(text: str, prefix: str = "    ") -> str:
    lines = str(text or "").splitlines()
    if not lines:
        return prefix.rstrip()
    return "\n".join((prefix + line) if line else prefix.rstrip() for line in lines)




def _parse_step0_verdict_payload(payload: Any) -> Dict[str, Any]:
    def _default() -> Dict[str, Any]:
        return {
            "verdict": "C",
            "direct_sufficient": None,
            "should_use_tools": None,
            "confidence": None,
            "rationale": "",
            "missing_elements": [],
            "explicit_errors": [],
            "uncertainties": [],
        }

    def _extract_float_from_raw(raw: str, key: str) -> Optional[float]:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw or "")
        if not m:
            return None
        try:
            v = float(m.group(1))
        except Exception:
            return None
        return max(0.0, min(1.0, v))

    def _extract_string_from_raw(raw: str, key: str) -> str:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)', raw or "", re.S)
        return m.group(1).strip() if m else ""

    def _extract_bool_from_raw(raw: str, key: str) -> Optional[bool]:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false)', raw or "", re.I)
        if not m:
            return None
        return m.group(1).lower() == "true"

    if not isinstance(payload, dict):
        return _default()

    # Fast path: fully parsed JSON object.
    if not payload.get("_parse_error"):
        verdict = _parse_letter(str(payload.get("verdict") or ""), "ABC")
        conf_raw = payload.get("confidence")
        try:
            confidence = float(conf_raw)
        except Exception:
            confidence = None
        if confidence is not None:
            confidence = max(0.0, min(1.0, confidence))
        direct_sufficient = payload.get("direct_sufficient")
        if not isinstance(direct_sufficient, bool):
            direct_sufficient = (verdict == "A")
        should_use_tools = payload.get("should_use_tools")
        if not isinstance(should_use_tools, bool):
            should_use_tools = (verdict != "A")
        return {
            "verdict": verdict,
            "direct_sufficient": direct_sufficient,
            "should_use_tools": should_use_tools,
            "confidence": confidence,
            "rationale": str(payload.get("rationale") or "").strip(),
            "missing_elements": _coerce_str_list(payload.get("missing_elements")),
            "explicit_errors": _coerce_str_list(payload.get("explicit_errors")),
            "uncertainties": _coerce_str_list(payload.get("uncertainties")),
        }

    # Fallback path: salvage key fields from truncated raw text.
    raw = str(payload.get("_raw") or "")
    if not raw:
        return _default()
    verdict = "C"
    m = re.search(r'"verdict"\s*:\s*"([ABC])"', raw)
    if m:
        verdict = _parse_letter(m.group(1), "ABC")
    confidence = _extract_float_from_raw(raw, "confidence")
    rationale = _extract_string_from_raw(raw, "rationale")
    direct_sufficient = _extract_bool_from_raw(raw, "direct_sufficient")
    should_use_tools = _extract_bool_from_raw(raw, "should_use_tools")
    if direct_sufficient is None:
        direct_sufficient = (verdict == "A")
    if should_use_tools is None:
        should_use_tools = (verdict != "A")
    return {
        "verdict": verdict,
        "direct_sufficient": direct_sufficient,
        "should_use_tools": should_use_tools,
        "confidence": confidence,
        "rationale": rationale,
        "missing_elements": [],
        "explicit_errors": [],
        "uncertainties": [],
    }


def _step0_normalize_issue_list(v: Any, *, limit: int = 3) -> List[str]:
    out: List[str] = []
    for s in _coerce_str_list(v):
        t = re.sub(r"\s+", " ", s).strip(" -;；,，。")
        if not t:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= max(1, limit):
            break
    return out


def _question_complexity_flags(question: str) -> Dict[str, bool]:
    q = str(question or "")
    q_lower = q.lower()
    multi_clause = sum(q.count(x) for x in ["；", ";", "、"]) >= 2 or len(re.findall(r"\$[^$]+\$", q)) >= 3
    multi_part = bool(re.search(r"(分别|比较|并|以及|同时|给出.*和.*|讨论.*与.*)", q))
    derivation_heavy = bool(re.search(r"(推导|证明|证明出|写出|给出表达式|说明适用条件|解释原因|由.*推出)", q))
    grounding_sensitive = bool(re.search(r"(RPA|SCFT|spinodal|binodal|Donnan|Cahn|Hilliard|reptation|tube model|OSF|WLC|Flory[–-]Huggins)", q, re.I))
    topology_sensitive = bool(re.search(r"(星型|环状|嵌段|网络|刷|接枝|受限|拓扑|多臂|rod|coil)", q, re.I))
    return {
        "multi_clause": multi_clause,
        "multi_part": multi_part,
        "derivation_heavy": derivation_heavy,
        "grounding_sensitive": grounding_sensitive,
        "topology_sensitive": topology_sensitive,
    }


def _calibrate_step0_gate(payload: Dict[str, Any], *, question: str = "") -> Dict[str, Any]:
    original_verdict = _parse_letter(str(payload.get("verdict") or ""), "ABC")
    direct_sufficient = payload.get("direct_sufficient")
    if not isinstance(direct_sufficient, bool):
        direct_sufficient = (original_verdict == "A")
    should_use_tools = payload.get("should_use_tools")
    if not isinstance(should_use_tools, bool):
        should_use_tools = (original_verdict != "A")

    confidence = payload.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except Exception:
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))

    missing_elements = _step0_normalize_issue_list(payload.get("missing_elements"), limit=2)
    explicit_errors = _step0_normalize_issue_list(payload.get("explicit_errors"), limit=2)
    uncertainties = _step0_normalize_issue_list(payload.get("uncertainties"), limit=2)

    flags = _question_complexity_flags(question)
    hard_block = bool(explicit_errors)
    soft_issue_count = len(missing_elements) + len(uncertainties)

    # Thresholds tuned from prior M3 runs:
    # - phase1 was too strict and collapsed into near-M1
    # - phase3 achieved real split, but many early-exit samples were still judged as "should_use_tools=true"
    # Therefore, final early-exit requires BOTH answer sufficiency and low expected retrieval utility.
    conf_threshold = 0.88
    if flags["derivation_heavy"] or flags["grounding_sensitive"] or flags["multi_part"]:
        conf_threshold = 0.91
    if flags["topology_sensitive"] and flags["multi_clause"]:
        conf_threshold = 0.93

    gate_reason = "keep_original"
    route_action = "retrieve"
    final_verdict = "C"

    if hard_block:
        final_verdict = "B"
        gate_reason = "retrieve_due_to_explicit_errors"
        route_action = "retrieve"
    elif not direct_sufficient:
        final_verdict = "C"
        gate_reason = "retrieve_due_to_not_direct_sufficient"
        route_action = "retrieve"
    elif should_use_tools:
        final_verdict = "C"
        gate_reason = "retrieve_due_to_positive_tool_utility"
        route_action = "retrieve"
    elif confidence is None or confidence < conf_threshold:
        final_verdict = "C"
        gate_reason = "retrieve_due_to_low_confidence"
        route_action = "retrieve"
    elif soft_issue_count > 0:
        final_verdict = "C"
        gate_reason = "retrieve_due_to_open_issues"
        route_action = "retrieve"
    else:
        final_verdict = "A"
        gate_reason = "early_exit_confident_and_low_utility"
        route_action = "early_exit"

    return {
        "original_verdict": original_verdict,
        "final_verdict": final_verdict,
        "direct_sufficient": direct_sufficient,
        "should_use_tools": should_use_tools,
        "confidence": confidence,
        "rationale": str(payload.get("rationale") or "").strip(),
        "missing_elements": missing_elements,
        "explicit_errors": explicit_errors,
        "uncertainties": uncertainties,
        "gate_reason": gate_reason,
        "route_action": route_action,
        "question_flags": flags,
        "confidence_threshold_used": conf_threshold,
    }

def _strip_heading_tags(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"【\s*解题思路.*?】", "", s)
    s = re.sub(r"【\s*最终答案\s*】", "", s)
    return s.strip()


def _split_answer_sections(answer_text: str) -> Tuple[str, str]:
    ans = (answer_text or "").strip()
    m1 = re.search(r"【\s*解题思路.*?】", ans)
    m2 = re.search(r"【\s*最终答案\s*】", ans)
    if m1 and m2 and m2.start() > m1.end():
        return ans[m1.end():m2.start()].strip(), ans[m2.end():].strip()
    return ans, ans


def _count_tag(s: str, tag: str) -> int:
    return (s or "").count(tag)


def _count_unescaped_dollars(s: str) -> int:
    return len(re.findall(r"(?<!\\)\$", s or ""))


def _extract_level1_bracket_headings(s: str) -> List[str]:
    hs: List[str] = []
    for line in (s or "").splitlines():
        t = line.strip()
        if re.fullmatch(r"【[^】\n]{1,80}】", t):
            hs.append(t)
    return hs


def _looks_truncated(s: str) -> bool:
    s = (s or "").rstrip()
    if not s:
        return True
    tail = s[-120:]
    if "Correction:" in tail or "V0/" in tail:
        return True
    if re.search(r"(=|/|\\|\(|\[|\{|:|\-|\*)\s*$", s):
        if not s.endswith(":\n") and not s.endswith("：\n"):
            return True
    return False


def _answer_format_errors(ans: str) -> List[str]:
    a = (ans or "").strip()
    if not a:
        return ["empty"]
    errors: List[str] = []
    r_tag = "【解题思路与公式推导思路】"
    f_tag = "【最终答案】"
    r_cnt = _count_tag(a, r_tag)
    f_cnt = _count_tag(a, f_tag)
    if r_cnt == 0:
        errors.append("missing_reasoning_heading")
    elif r_cnt > 1:
        errors.append("duplicate_reasoning_heading")
    if f_cnt == 0:
        errors.append("missing_final_heading")
    elif f_cnt > 1:
        errors.append("duplicate_final_heading")
    hs = _extract_level1_bracket_headings(a)
    extras = [h for h in hs if h not in {r_tag, f_tag}]
    if extras:
        errors.append("extra_level1_heading")
    r_i = a.find(r_tag)
    f_i = a.find(f_tag)
    if r_i != -1 and f_i != -1 and f_i < r_i:
        errors.append("tag_order_wrong")
    if r_i != -1 and f_i != -1 and f_i > r_i:
        reasoning = a[r_i + len(r_tag):f_i].strip()
        final = a[f_i + len(f_tag):].strip()
        if not reasoning:
            errors.append("empty_reasoning_section")
        if not final:
            errors.append("empty_final_section")
    if _has_unclosed_code_fence(a):
        errors.append("unclosed_code_fence")
    if "Correction:" in a:
        errors.append("contains_correction_marker")
    if _looks_truncated(a):
        errors.append("looks_truncated")
    if (_count_unescaped_dollars(a) % 2) == 1:
        errors.append("unbalanced_math_delimiter")
    out: List[str] = []
    seen = set()
    for e in errors:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def _normalize_two_section_output(raw: str) -> str:
    t = (raw or "").strip()
    if not t:
        return "【解题思路与公式推导思路】\n\n【最终答案】"
    reasoning, final = _split_answer_sections(t)
    reasoning = _strip_heading_tags(reasoning)
    final = _strip_heading_tags(final)
    body = _strip_heading_tags(t)
    if not reasoning:
        reasoning = body
    if not final:
        final = body
    return (
        "【解题思路与公式推导思路】\n" + reasoning.strip() + "\n\n"
        + "【最终答案】\n" + final.strip()
    ).strip()


def _has_unclosed_code_fence(s: str) -> bool:
    return (len(re.findall(r"```", s or "")) % 2) == 1


def _integrity_issues(s: str) -> List[str]:
    s = s or ""
    out: List[str] = []
    if not s.strip():
        out.append("empty")
        return out
    if _has_unclosed_code_fence(s):
        out.append("unclosed_code_fence")
    if "Correction:" in s:
        out.append("correction_tail")
    if s.count("【解题思路与公式推导思路】") > 1:
        out.append("duplicate_reasoning_heading")
    if s.count("【最终答案】") > 1:
        out.append("duplicate_final_heading")
    if _looks_truncated(s):
        out.append("looks_truncated")
    for e in _answer_format_errors(s):
        if e not in out:
            out.append(e)
    if len(s.strip()) < 20:
        out.append("too_short")
    return out


def _is_severely_broken(issues: List[str]) -> bool:
    severe = {
        "empty", "unclosed_code_fence", "looks_truncated", "tag_order_wrong",
        "missing_reasoning_heading", "missing_final_heading",
        "duplicate_reasoning_heading", "duplicate_final_heading",
        "extra_level1_heading", "empty_final_section", "unbalanced_math_delimiter",
    }
    return any(x in severe for x in issues)


def _truncate_chars(s: str, max_chars: int) -> str:
    s = str(s or "")
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars]


def normalize_book_title(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def make_chapter_label(book_title: str, chapter_no: int, chapter_title: str) -> str:
    return f"《{normalize_book_title(book_title)}》第{int(chapter_no)}章：{(chapter_title or '').strip() or f'chapter {int(chapter_no)}'}"


def make_section_label(chapter_label: str, section_title: str) -> str:
    return f"{chapter_label}｜节：{(section_title or '').strip()}"


def _coerce_int_list(v: Any, *, min_n: int, max_n: int, allowed: Optional[List[int]] = None) -> List[int]:
    out: List[int] = []
    if isinstance(v, list):
        for x in v:
            try:
                xi = int(x)
            except Exception:
                continue
            if allowed is not None and xi not in allowed:
                continue
            if xi not in out:
                out.append(xi)
    return out[:max_n]


def _coerce_pairs(v: Any, *, max_n: int, allowed_books: Optional[List[int]] = None, allowed_chapters: Optional[Dict[int, List[int]]] = None) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if isinstance(v, list):
        for row in v:
            if isinstance(row, dict):
                b = row.get("book_id")
                c = row.get("chapter_no")
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                b, c = row[0], row[1]
            else:
                continue
            try:
                b = int(b)
                c = int(c)
            except Exception:
                continue
            if allowed_books is not None and b not in allowed_books:
                continue
            if allowed_chapters is not None and c not in allowed_chapters.get(b, []):
                continue
            pair = (b, c)
            if pair not in out:
                out.append(pair)
    return out[:max_n]


def _coerce_triplets(v: Any, *, max_n: int, allowed_pairs: Optional[Dict[Tuple[int, int], List[str]]] = None) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    if isinstance(v, list):
        for row in v:
            if isinstance(row, dict):
                b = row.get("book_id")
                c = row.get("chapter_no")
                s = row.get("section_id")
            elif isinstance(row, (list, tuple)) and len(row) >= 3:
                b, c, s = row[0], row[1], row[2]
            else:
                continue
            try:
                b = int(b)
                c = int(c)
                s = str(s)
            except Exception:
                continue
            if allowed_pairs is not None and s not in allowed_pairs.get((b, c), []):
                continue
            item = (b, c, s)
            if item not in out:
                out.append(item)
    return out[:max_n]


def _auto(label: str) -> int:
    return {
        "direct": 8192,
        "final": 8192,
        "post_format": 12000,
        "rank": 1024,
        "judge": 64,
        "compare": 64,
        "self_assess": 256,
    }.get(label, 2048)


@dataclass
class QAResult:
    question: str
    answer: str
    answer1: str
    answer2: str
    verdict0: str
    selected: str
    outline: str
    audit_outline: str
    final_answer: str
    answer_raw: str
    report_question: str = ""
    report_retrieval_content: str = ""
    report_key_steps: str = ""
    report_solution_process: str = ""
    verdict0_confidence: Optional[float] = None
    verdict0_rationale: str = ""
    verdict0_missing_elements: List[str] = field(default_factory=list)
    verdict0_explicit_errors: List[str] = field(default_factory=list)
    verdict0_uncertainties: List[str] = field(default_factory=list)
    verdict0_gate_reason: str = ""
    verdict0_direct_sufficient: Optional[bool] = None
    verdict0_should_use_tools: Optional[bool] = None
    verdict0_route_action: str = ""
    verdict0_question_flags: Dict[str, bool] = field(default_factory=dict)
    verdict0_confidence_threshold_used: Optional[float] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)
    step1_candidates_books: List[Dict[str, Any]] = field(default_factory=list)
    step2_selected_books: List[Dict[str, Any]] = field(default_factory=list)
    step3_candidate_chapters: List[Dict[str, Any]] = field(default_factory=list)
    step4_selected_chapters: List[Dict[str, Any]] = field(default_factory=list)
    step5_selected_sections: List[Dict[str, Any]] = field(default_factory=list)
    section_views: List[Dict[str, Any]] = field(default_factory=list)
    final_sources: List[Dict[str, Any]] = field(default_factory=list)
    run_dir: str = ""
    # legacy aliases for eval compatibility
    book_ids: List[int] = field(default_factory=list)
    book_chapters: List[Tuple[int, int]] = field(default_factory=list)
    extracts: List[str] = field(default_factory=list)

    # testsys fields
    test_mode: str = ""
    executed_steps: List[str] = field(default_factory=list)
    retrieval_used: bool = False
    retrieval_rounds: int = 0
    forced_retrieval: bool = False
    no_retrieval: bool = False
    expert_prompt_used: bool = False
    chapter_only_mode: bool = False
    skip_section_ranking: bool = False
    topk_books: int = 0
    topk_books_view: int = 0
    topk_candidate_chapters: int = 0
    topk_selected_chapters: int = 0
    topk_sections: int = 0
    section_budget_used: int = 0
    step5_selection_mode: str = ""
    p_solve: Optional[float] = None
    p_solve_reason: str = ""
    duration_sec: float = 0.0
    step_durations_sec: Dict[str, float] = field(default_factory=dict)
    confidence_threshold: Optional[float] = None
    m5_gate_triggered_retrieval: Optional[bool] = None
    section_chars_total: int = 0
    step5_available_sections_total: int = 0
    step1_json_ok: Optional[bool] = None
    step2_json_ok: Optional[bool] = None
    step3_json_ok: Optional[bool] = None
    step4_json_ok: Optional[bool] = None
    step5_json_ok: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiBookQAAgent:
    def __init__(self, library: BookLibrary, cfg: Any):
        self.lib = library
        self.cfg = cfg

    def _model(self, key: str, default: str) -> str:
        try:
            qa_models = getattr(self.cfg, "qa_models")
            v = getattr(qa_models, key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            pass
        return default

    def _qa_val(self, key: str, default: Any) -> Any:
        try:
            qa = getattr(self.cfg, "qa")
            v = getattr(qa, key)
            return default if v is None else v
        except Exception:
            return default

    def _load_expert_prompt(self) -> str:
        p = str(self._qa_val("expert_prompt_path", "prompts/expert_polymer_simulation.md") or "prompts/expert_polymer_simulation.md")
        try:
            return Path(p).read_text(encoding="utf-8")
        except Exception:
            return ""

    def _repair_answer_format(
        self,
        session,
        *,
        question: str,
        raw_answer: str,
        errors: List[str],
        model: str,
        system_msgs: List[Dict[str, str]],
        max_tokens: int,
    ) -> str:
        prompt_fix = (
            "你是最终输出格式修复器。你的任务不是重写答案，只能做最小格式修复。\n"
            "只允许：补两个固定一级标题、删除多余一级标题、修复未闭合代码块、修复明显截断尾巴、把裸数学尽量包进 $...$ 或 $$...$$。\n"
            "禁止改变实质性的物理推导、公式含义、结论与条件。\n"
            "修复后必须满足：\n"
            "1. 只能出现且仅出现两个一级标题：\n"
            "   【解题思路与公式推导思路】\n"
            "   【最终答案】\n"
            "2. 不得出现其他一级标题。\n"
            "3. 两个标题都必须非空。\n"
            "4. 所有数学符号/公式必须放在 $...$ 或 $$...$$ 中。\n"
            "5. 不要输出解释，不要输出 JSON，只输出修复后的完整答案文本。\n\n"
            f"[问题]\n{question}\n\n"
            f"[检测到的格式错误]\n- " + "\n- ".join(errors) + "\n\n"
            f"[待修复答案]\n{raw_answer}"
        )
        return call_chat(session, model, system_msgs + [{"role": "user", "content": prompt_fix}], max_tokens=max_tokens, temperature=0.0).strip()

    def _enforce_answer_format(
        self,
        session,
        *,
        stage_name: str,
        question: str,
        raw_answer: str,
        generation_model: str,
        generation_prompt: str,
        repair_model: str,
        system_msgs: List[Dict[str, str]],
        generation_max_tokens: int,
        repair_max_tokens: int,
        trace: List[Dict[str, Any]],
    ) -> str:
        enabled = bool(self._qa_val("answer_format_validation_enabled", True))
        if not enabled:
            return _normalize_two_section_output(raw_answer)
        max_repairs = int(self._qa_val("answer_format_max_repairs", 1) or 1)
        max_reanswers = int(self._qa_val("answer_format_max_reanswers", 1) or 1)
        current = (raw_answer or "").strip()
        errors = _answer_format_errors(current)
        trace.append({"step": f"{stage_name}.format_check", "errors": list(errors), "ok": not errors})
        if not errors:
            return _normalize_two_section_output(current)
        for attempt in range(1, max(0, max_repairs) + 1):
            current = self._repair_answer_format(
                session,
                question=question,
                raw_answer=current,
                errors=errors,
                model=repair_model,
                system_msgs=system_msgs,
                max_tokens=repair_max_tokens,
            )
            errors = _answer_format_errors(current)
            trace.append({"step": f"{stage_name}.format_repair", "attempt": attempt, "errors": list(errors), "ok": not errors})
            if not errors:
                return _normalize_two_section_output(current)
        for attempt in range(1, max(0, max_reanswers) + 1):
            re_prompt = (
                generation_prompt
                + "\n\n[上一次输出存在的格式错误]\n- " + "\n- ".join(errors)
                + "\n\n请重新作答，严格遵守固定输出结构与数学排版要求。"
            )
            current = call_chat(session, generation_model, system_msgs + [{"role": "user", "content": re_prompt}], max_tokens=generation_max_tokens, temperature=0.2).strip()
            trace.append({"step": f"{stage_name}.reanswer", "attempt": attempt})
            errors = _answer_format_errors(current)
            trace.append({"step": f"{stage_name}.format_check_after_reanswer", "attempt": attempt, "errors": list(errors), "ok": not errors})
            if not errors:
                return _normalize_two_section_output(current)
            for rep in range(1, max(0, max_repairs) + 1):
                current = self._repair_answer_format(
                    session,
                    question=question,
                    raw_answer=current,
                    errors=errors,
                    model=repair_model,
                    system_msgs=system_msgs,
                    max_tokens=repair_max_tokens,
                )
                errors = _answer_format_errors(current)
                trace.append({"step": f"{stage_name}.format_repair_after_reanswer", "reanswer_attempt": attempt, "attempt": rep, "errors": list(errors), "ok": not errors})
                if not errors:
                    return _normalize_two_section_output(current)
        trace.append({"step": f"{stage_name}.format_fallback", "errors": list(errors)})
        return _normalize_two_section_output(current)

    def ask(
        self,
        session,
        question: str,
        *,
        out_dir: Optional[Path] = None,
        verbose: bool = True,
        mode: Optional[str] = None,
        topk_books_override: Optional[int] = None,
        topk_pairs_override: Optional[int] = None,
        topk_sections_override: Optional[int] = None,
    ) -> QAResult:
        q = (question or "").strip()
        if not q:
            raise ValueError("Empty question")

        t_all_start = time.perf_counter()
        step_durations_sec: Dict[str, float] = {}
        def _mark(name: str, t0: float) -> None:
            step_durations_sec[name] = round(step_durations_sec.get(name, 0.0) + (time.perf_counter() - t0), 6)

        llm_default = str(getattr(getattr(self.cfg, "models"), "llm_model"))
        direct_model = self._model("direct_answer_model", llm_default)
        judge_model = self._model("judge_model", llm_default)
        rank1_model = self._model("rank_books_stage1_model", self._model("rank_books_model", llm_default))
        rank2_model = self._model("rank_books_stage2_model", self._model("rank_books_model", llm_default))
        rank3_model = self._model("rank_chapters_stage3_model", self._model("rank_chapters_model", llm_default))
        rank4_model = self._model("rank_chapters_stage4_model", self._model("rank_chapters_model", llm_default))
        rank5_model = self._model("rank_sections_stage5_model", self._model("rank_chapters_model", llm_default))
        answer_model = self._model("final_answer_model", llm_default)
        compare_model = self._model("compare_model", llm_default)
        format_model = self._model("format_model", llm_default)
        self_assess_model = self._model("self_assess_model", llm_default)

        direct_max = int(self._qa_val("direct_answer_max_tokens", _auto("direct")) or _auto("direct"))
        judge_max = int(self._qa_val("judge_max_tokens", _auto("judge")) or _auto("judge"))
        rank_max = int(self._qa_val("rank_max_tokens", _auto("rank")) or _auto("rank"))
        final_max = int(self._qa_val("final_answer_max_tokens", _auto("final")) or _auto("final"))
        compare_max = int(self._qa_val("answer_compare_max_tokens", _auto("compare")) or _auto("compare"))
        post_format_max = int(self._qa_val("post_format_max_tokens", _auto("post_format")) or _auto("post_format"))
        answer_format_repair_max = int(self._qa_val("answer_format_repair_max_tokens", post_format_max) or post_format_max)
        self_assess_max = int(self._qa_val("self_assess_max_tokens", _auto("self_assess")) or _auto("self_assess"))

        step1_min = int(self._qa_val("step1_book_candidates_min", 3))
        step1_max = int(self._qa_val("step1_book_candidates_max", 7))
        step2_min = int(self._qa_val("step2_books_to_view_min", 1))
        step2_max = int(self._qa_val("step2_books_to_view_max", 2))
        step3_max = int(self._qa_val("step3_chapters_per_book_max", 5))
        step4_max = int(self._qa_val("step4_chapters_to_view_max", 2))
        step5_sections_total_max = int(self._qa_val("step5_sections_total_max", 10))
        step5_sections_per_chapter_max = int(self._qa_val("step5_sections_per_chapter_max", 6))
        allow_masked_sections = bool(self._qa_val("allow_masked_sections", False))
        prefer_exposure = list(self._qa_val("prefer_exposure", ["expose", "caution"]))
        min_section_quality = float(self._qa_val("min_section_quality_score", 0.0) or 0.0)
        section_text_max_chars = int(self._qa_val("section_text_max_chars", 80000))
        sections_total_max_chars = int(self._qa_val("sections_total_max_chars", 240000))
        confidence_threshold = float(self._qa_val("confidence_threshold", 0.7) or 0.7)

        mode0 = (mode or self._qa_val("test_mode", "M3") or "M3").strip().upper()
        if mode0.startswith("M8"):
            mode0 = "M8"

        topk_books = int(topk_books_override if topk_books_override is not None else self._qa_val("topk_books", step1_max) or step1_max)
        topk_books_view = int(self._qa_val("topk_books_view", step2_max) or step2_max)
        topk_candidate_chapters = int(self._qa_val("topk_candidate_chapters", step3_max) or step3_max)
        topk_selected_chapters = int(self._qa_val("topk_selected_chapters", step4_max) or step4_max)
        if topk_sections_override is None and topk_pairs_override is not None and mode0 == "M8":
            topk_sections_override = topk_pairs_override
        topk_sections = int(topk_sections_override if topk_sections_override is not None else self._qa_val("topk_sections", step5_sections_total_max) or step5_sections_total_max)
        if mode0 == "M8":
            step5_sections_total_max = max(1, topk_sections)
        else:
            topk_sections = step5_sections_total_max

        no_retrieval = mode0 in {"M2", "M4"}
        # M9 = 专家提示词 + 强制查阅 library（M4 与 M1 的组合模式）
        expert_prompt_used = mode0 in {"M4", "M9"} or bool(self._qa_val("use_expert_prompt", False))
        forced_retrieval = mode0 in {"M1", "M6", "M7", "M8", "M9"}
        chapter_only_mode = mode0 == "M6"
        skip_section_ranking = mode0 == "M7"
        allow_early_exit = mode0 == "M3"
        p_solve: Optional[float] = None
        p_solve_reason = ""
        m5_gate_triggered_retrieval: Optional[bool] = None

        system_msgs: List[Dict[str, str]] = []
        if expert_prompt_used:
            exp = self._load_expert_prompt().strip()
            if exp:
                system_msgs.append({"role": "system", "content": exp})

        trace: List[Dict[str, Any]] = []
        executed_steps: List[str] = []
        def _log(step: str, msg: str) -> None:
            trace.append({"step": step, "message": msg})
            if verbose:
                print(f"[{step}] {msg}")

        # Step0
        t0 = time.perf_counter()
        executed_steps.append("Step0")
        _log("Step0", "直接回答问题")
        prompt0 = (
            "请直接回答下面这个问题，并严格按如下格式输出：\n"
            "【解题思路与公式推导思路】\n"
            "- 用要点列出解题路线；涉及公式时给出关键推导链条。\n"
            "【最终答案】\n"
            "- 给出最终结论/表达式/数值结果（尽量简洁）。\n\n"
            "要求：所有数学符号、变量、公式必须放在 $...$ 或 $$...$$ 中；不得伪造书籍/章节来源。\n\n"
            f"[问题]\n{q}\n"
        )
        answer1_raw = call_chat(session, direct_model, system_msgs + [{"role": "user", "content": prompt0}], max_tokens=direct_max, temperature=0.2).strip()
        answer1 = self._enforce_answer_format(
            session,
            stage_name="Step0",
            question=q,
            raw_answer=answer1_raw,
            generation_model=direct_model,
            generation_prompt=prompt0,
            repair_model=format_model,
            system_msgs=system_msgs,
            generation_max_tokens=direct_max,
            repair_max_tokens=answer_format_repair_max,
            trace=trace,
        )
        trace.append({"step": "Step0.answer1", "answer1": answer1})
        _mark("Step0", t0)

        verdict0 = "C"
        verdict0_confidence: Optional[float] = None
        verdict0_rationale = ""
        verdict0_missing_elements: List[str] = []
        verdict0_explicit_errors: List[str] = []
        verdict0_uncertainties: List[str] = []
        verdict0_direct_sufficient: Optional[bool] = None
        verdict0_should_use_tools: Optional[bool] = None
        verdict0_route_action = ""
        verdict0_question_flags: Dict[str, bool] = {}
        verdict0_confidence_threshold_used: Optional[float] = None
        if allow_early_exit:
            tj = time.perf_counter()
            jprompt = (
                "你是一个严格、客观、中立的 Step0 自适应检索路由器。你的任务不是判断答案‘像不像对’，而是同时判断两件事："
                "（1）当前直答是否已足够可靠可直接交付；（2）继续检索是否仍有明显收益。\n\n"
                "请输出 ONLY JSON：\n"
                "{\"verdict\": \"A|B|C\", \"direct_sufficient\": true, \"should_use_tools\": false, \"confidence\": 0.0, \"rationale\": \"不超过40字\", \"missing_elements\": [\"最多2项\"], \"explicit_errors\": [\"最多2项\"], \"uncertainties\": [\"最多2项\"]}\n\n"
                "字段含义：\n"
                "- verdict：A=当前答案足够好且继续检索收益低；B=当前答案存在明确实质错误；C=当前答案尚不宜直接交付，或继续检索仍有明显收益。\n"
                "- direct_sufficient：不查资料的前提下，这个答案是否已足够正确、完整、可直接交付。\n"
                "- should_use_tools：即使当前答案大体可用，继续查阅书/章/节是否仍有明显预期收益。\n"
                "- confidence：你对本次路由判断本身的把握。\n"
                "- missing_elements / explicit_errors / uncertainties：只填写会实质性影响是否应停止检索的要点。\n\n"
                "判定规则：\n"
                "- 只有在 direct_sufficient=true 且 should_use_tools=false 时，verdict 才能为 A。\n"
                "- 若存在明确且实质性的错误，verdict 置为 B。\n"
                "- 其余情况一律为 C。特别是：答案大体可答，但继续检索仍可能显著提升正确性、完整性、grounding 或边界条件把握时，也必须为 C。\n\n"
                "强约束：\n"
                "- 多小问、长推导、特殊拓扑/受限几何、需区分相近判据、需明确适用条件的题目，若继续检索仍可能显著降低混淆风险，应令 should_use_tools=true。\n"
                "- 不要因为答案像教科书风格就给 A。\n"
                "- 也不要因为还能补充非关键背景就机械地给 C。\n"
                "- 只依据【问题】与【候选答案】本身判断，不得脑补未写出的关键步骤。\n\n"
                f"[问题]\n{q}\n\n[候选答案]\n{answer1}\n"
            )
            out0 = call_json(session, judge_model, system_msgs + [{"role": "user", "content": jprompt}], max_tokens=max(judge_max, 320), temperature=0.0)
            verdict0_payload = _parse_step0_verdict_payload(out0)
            gate0 = _calibrate_step0_gate(verdict0_payload, question=q)
            verdict0 = gate0["final_verdict"]
            verdict0_direct_sufficient = gate0["direct_sufficient"]
            verdict0_should_use_tools = gate0["should_use_tools"]
            verdict0_confidence = gate0["confidence"]
            verdict0_rationale = gate0["rationale"]
            verdict0_missing_elements = gate0["missing_elements"]
            verdict0_explicit_errors = gate0["explicit_errors"]
            verdict0_uncertainties = gate0["uncertainties"]
            verdict0_route_action = gate0["route_action"]
            verdict0_question_flags = gate0["question_flags"]
            verdict0_confidence_threshold_used = gate0["confidence_threshold_used"]
            trace.append({
                "step": "Step0.verdict",
                "raw": out0,
                "original_verdict": gate0["original_verdict"],
                "verdict": verdict0,
                "direct_sufficient": verdict0_direct_sufficient,
                "should_use_tools": verdict0_should_use_tools,
                "confidence": verdict0_confidence,
                "rationale": verdict0_rationale,
                "missing_elements": verdict0_missing_elements,
                "explicit_errors": verdict0_explicit_errors,
                "uncertainties": verdict0_uncertainties,
                "gate_reason": gate0["gate_reason"],
                "route_action": verdict0_route_action,
                "question_flags": verdict0_question_flags,
                "confidence_threshold_used": verdict0_confidence_threshold_used,
            })
            _mark("Judge0", tj)

        if mode0 == "M5":
            ts = time.perf_counter()
            sprompt = (
                "只根据下面的问题，评估你在不查阅任何书籍/章节的情况下独立正确作答的把握。"
                "输出 ONLY JSON: {\"p_solve\": 0.0, \"reason\": \"简短理由\"}.\n\n"
                f"[问题]\n{q}\n"
            )
            out_s = call_json(session, self_assess_model, system_msgs + [{"role": "user", "content": sprompt}], max_tokens=self_assess_max, temperature=0.0)
            try:
                p_solve = float(out_s.get("p_solve"))
            except Exception:
                p_solve = None
            p_solve_reason = str(out_s.get("reason") or "") if isinstance(out_s, dict) else ""
            m5_gate_triggered_retrieval = None if p_solve is None else (p_solve < confidence_threshold)
            trace.append({"step": "M5.self_assess", "raw": out_s, "p_solve": p_solve, "reason": p_solve_reason})
            _mark("SelfAssess", ts)
            if p_solve is not None and p_solve >= confidence_threshold:
                no_retrieval = True
                forced_retrieval = False
            else:
                no_retrieval = False
                forced_retrieval = True

        def _finalize_no_retrieval(selected_name: str = "answer1") -> QAResult:
            retrieval_context = {
                "step1_candidates_books": [],
                "step2_selected_books": [],
                "step3_candidate_chapters": [],
                "step4_selected_chapters": [],
                "step5_selected_sections": [],
                "final_sources": [],
                "section_views": [],
            }
            fmt = self._post_format_answer(session, q, answer1, retrieval_context=retrieval_context, model=format_model, max_tokens=post_format_max)
            report_fields = self._build_structured_report_fields(question=q, raw_answer=answer1, final_reasoning=fmt['final_reasoning'], retrieval_context=retrieval_context)
            final = f"【解题思路与公式推导思路】\n{fmt['final_reasoning']}\n\n【最终答案】\n{fmt['final_answer']}".strip()
            duration_sec = round(time.perf_counter() - t_all_start, 6)
            res = QAResult(
                question=q,
                answer=final,
                answer1=answer1,
                answer2="",
                verdict0=verdict0,
                verdict0_confidence=verdict0_confidence,
                verdict0_rationale=verdict0_rationale,
                verdict0_missing_elements=verdict0_missing_elements,
                verdict0_explicit_errors=verdict0_explicit_errors,
                verdict0_uncertainties=verdict0_uncertainties,
                verdict0_gate_reason=(next((x.get("gate_reason", "") for x in reversed(trace) if x.get("step") == "Step0.verdict"), "")),
                verdict0_direct_sufficient=verdict0_direct_sufficient,
                verdict0_should_use_tools=verdict0_should_use_tools,
                verdict0_route_action=verdict0_route_action,
                verdict0_question_flags=verdict0_question_flags,
                verdict0_confidence_threshold_used=verdict0_confidence_threshold_used,
                selected=selected_name,
                outline=fmt["final_reasoning"],
                audit_outline=fmt["audit_reasoning"],
                final_answer=fmt["final_answer"],
                answer_raw=answer1,
                report_question=report_fields["report_question"],
                report_retrieval_content=report_fields["report_retrieval_content"],
                report_key_steps=report_fields["report_key_steps"],
                report_solution_process=report_fields["report_solution_process"],
                trace=trace,
                step1_candidates_books=[],
                step2_selected_books=[],
                step3_candidate_chapters=[],
                step4_selected_chapters=[],
                step5_selected_sections=[],
                section_views=[],
                final_sources=[],
                run_dir=str(out_dir or ""),
                book_ids=[],
                book_chapters=[],
                extracts=[],
                test_mode=mode0,
                executed_steps=executed_steps,
                retrieval_used=False,
                retrieval_rounds=0,
                forced_retrieval=bool(forced_retrieval),
                no_retrieval=True,
                expert_prompt_used=expert_prompt_used,
                chapter_only_mode=chapter_only_mode,
                skip_section_ranking=skip_section_ranking,
                topk_books=topk_books,
                topk_books_view=topk_books_view,
                topk_candidate_chapters=topk_candidate_chapters,
                topk_selected_chapters=topk_selected_chapters,
                topk_sections=topk_sections,
                section_budget_used=0,
                step5_selection_mode="none",
                p_solve=p_solve,
                p_solve_reason=p_solve_reason,
                duration_sec=duration_sec,
                step_durations_sec=step_durations_sec,
                confidence_threshold=(confidence_threshold if mode0 == "M5" else None),
                m5_gate_triggered_retrieval=m5_gate_triggered_retrieval,
                section_chars_total=0,
            )
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "qa_result.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            return res

        if no_retrieval:
            return _finalize_no_retrieval("answer1")
        if allow_early_exit and verdict0_route_action == "early_exit" and verdict0 == "A" and not forced_retrieval:
            trace.append({"step": "Step0.early_exit", "reason": "judge_A_low_utility"})
            return _finalize_no_retrieval("answer1")

        # Retrieval chain starts
        step1_candidates_books: List[Dict[str, Any]] = []
        step2_selected_books: List[Dict[str, Any]] = []
        step3_candidate_chapters: List[Dict[str, Any]] = []
        step4_selected_chapters: List[Dict[str, Any]] = []
        step5_selected_sections: List[Dict[str, Any]] = []
        section_views: List[Dict[str, Any]] = []
        final_sources: List[Dict[str, Any]] = []
        legacy_extracts: List[str] = []
        retrieval_used = True
        step1_json_ok = step2_json_ok = step3_json_ok = step4_json_ok = step5_json_ok = None

        entries = self.lib.load_index()
        allowed_books_all = [e.book_id for e in entries]

        # Step1 candidate books
        t = time.perf_counter()
        executed_steps.append("Step1")
        _log("Step1", "筛选候选书")
        summarybook = self.lib.read_summarybook()
        prompt1 = (
            f"根据下面的问题与书籍摘要，选出最相关的 {step1_min}~{topk_books} 本书。\n"
            "输出 ONLY JSON: {\"candidates\":[book_id,...],\"rationales\":{\"book_id\":\"简短理由\"}}。\n"
            f"book_id 只能从给定摘要中的编号选择。\n\n[问题]\n{q}\n\n[书籍摘要]\n{summarybook}"
        )
        out1 = call_json(session, rank1_model, system_msgs + [{"role": "user", "content": prompt1}], max_tokens=rank_max, temperature=0.0)
        step1_json_ok = isinstance(out1, dict) and isinstance(out1.get("candidates"), list)
        cand_books = _coerce_int_list(out1.get("candidates") if isinstance(out1, dict) else None, min_n=step1_min, max_n=topk_books, allowed=allowed_books_all)
        if len(cand_books) < step1_min:
            cand_books = allowed_books_all[:max(step1_min, min(topk_books, len(allowed_books_all)))]
        rat_map = out1.get("rationales") if isinstance(out1, dict) and isinstance(out1.get("rationales"), dict) else {}
        for bid in cand_books:
            e = self.lib.get_entry(bid)
            if not e:
                continue
            step1_candidates_books.append({"book_id": bid, "title": e.title, "rationale": str(rat_map.get(str(bid)) or rat_map.get(bid) or "")})
        trace.append({"step": "Step1.raw", "raw": out1, "selected": step1_candidates_books})
        _mark("Step1", t)

        # Step2 selected books (1-2)
        t = time.perf_counter()
        executed_steps.append("Step2")
        _log("Step2", "从候选书中决定深入查看的书")
        prompt2 = (
            f"根据下面的问题，从候选书中选出最值得深入查看的 {step2_min}~{topk_books_view} 本书。\n"
            "输出 ONLY JSON: {\"selected\":[book_id,...],\"why\":{\"book_id\":\"简短理由\"}}。\n"
            f"book_id 只能从 {cand_books} 中选择。\n\n[问题]\n{q}\n\n[候选书]\n" + "\n\n".join([
                f"book_id={x['book_id']}\ntitle={x['title']}\nrationale={x['rationale']}" for x in step1_candidates_books
            ])
        )
        out2 = call_json(session, rank2_model, system_msgs + [{"role": "user", "content": prompt2}], max_tokens=rank_max, temperature=0.0)
        step2_json_ok = isinstance(out2, dict) and isinstance(out2.get("selected"), list)
        selected_books = _coerce_int_list(out2.get("selected") if isinstance(out2, dict) else None, min_n=step2_min, max_n=topk_books_view, allowed=cand_books)
        if len(selected_books) < step2_min:
            selected_books = cand_books[:max(step2_min, min(topk_books_view, len(cand_books)))]
        why2 = out2.get("why") if isinstance(out2, dict) and isinstance(out2.get("why"), dict) else {}
        for bid in selected_books:
            e = self.lib.get_entry(bid)
            if e:
                step2_selected_books.append({"book_id": bid, "title": e.title, "why": str(why2.get(str(bid)) or why2.get(bid) or "")})
        trace.append({"step": "Step2.raw", "raw": out2, "selected": step2_selected_books})
        _mark("Step2", t)

        # Step3 candidate chapters
        t = time.perf_counter()
        executed_steps.append("Step3")
        _log("Step3", "为深入查看的书生成候选章")
        allowed_chapters: Dict[int, List[int]] = {}
        chapter_lookup: Dict[Tuple[int, int], Any] = {}
        chapter_section_counts: Dict[Tuple[int, int], int] = {}
        chapter_blocks: List[str] = []
        max_candidates_total = 0
        for bid in selected_books:
            e = self.lib.get_entry(bid)
            chapters = self.lib.list_chapters(bid, filter_policy={"prefer_exposure": ["expose", "caution", "masked", "unknown", ""]})
            allowed_chapters[bid] = [c.chapter_no for c in chapters]
            max_candidates_total += min(topk_candidate_chapters, len(chapters))
            for c in chapters:
                chapter_lookup[(bid, c.chapter_no)] = c
                chapter_blocks.append(
                    f"book_id={bid}\nchapter_no={c.chapter_no}\nchapter_label={make_chapter_label(e.title if e else f'book{bid}', c.chapter_no, c.title)}\nsummary={c.summary}\nkeywords={', '.join(c.keywords)}\nexposure={c.chapter_exposure_decision}\nquality={c.quality_label}:{c.quality_score}"
                )
        if max_candidates_total <= 0:
            raise ValueError("No chapter candidates available from selected books")
        prompt3 = (
            f"根据下面的问题与候选章信息，选出最相关的候选章。\n"
            f"要求：每本书最多选择 {topk_candidate_chapters} 章；总数不超过 {max_candidates_total}。\n"
            "输出 ONLY JSON: {\"candidates\":[[book_id, chapter_no], ...], \"why\": {\"book_id:chapter_no\": \"理由\"}}。\n"
            f"book_id 只能从 {selected_books} 中选择，章号必须来自给定候选章。\n\n[问题]\n{q}\n\n[候选章]\n" + "\n\n".join(chapter_blocks)
        )
        out3 = call_json(session, rank3_model, system_msgs + [{"role": "user", "content": prompt3}], max_tokens=rank_max, temperature=0.0)
        step3_json_ok = isinstance(out3, dict) and isinstance(out3.get("candidates"), list)
        raw_pairs = _coerce_pairs(out3.get("candidates") if isinstance(out3, dict) else None, max_n=max(1, len(chapter_blocks)), allowed_books=selected_books, allowed_chapters=allowed_chapters)
        cand_pairs: List[Tuple[int, int]] = []
        per_book_counts: Dict[int, int] = {}
        for bid, chno in raw_pairs:
            if per_book_counts.get(bid, 0) >= topk_candidate_chapters:
                continue
            cand_pairs.append((bid, chno))
            per_book_counts[bid] = per_book_counts.get(bid, 0) + 1
            if len(cand_pairs) >= max_candidates_total:
                break
        if not cand_pairs:
            for bid in selected_books:
                count = 0
                for ch in allowed_chapters.get(bid, []):
                    cand_pairs.append((bid, ch))
                    count += 1
                    if count >= topk_candidate_chapters or len(cand_pairs) >= max_candidates_total:
                        break
                if len(cand_pairs) >= max_candidates_total:
                    break
        if not chapter_only_mode:
            usable = [pair for pair in cand_pairs if chapter_section_counts.get(pair, 0) > 0]
            if usable:
                cand_pairs = usable
        why3 = out3.get("why") if isinstance(out3, dict) and isinstance(out3.get("why"), dict) else {}
        for bid, chno in cand_pairs:
            e = self.lib.get_entry(bid)
            cm = chapter_lookup.get((bid, chno))
            if not cm:
                continue
            step3_candidate_chapters.append({
                "book_id": bid,
                "chapter_no": chno,
                "chapter_title": cm.title,
                "chapter_label": make_chapter_label(e.title if e else f'book{bid}', chno, cm.title),
                "chapter_summary": cm.summary,
                "chapter_keywords": list(cm.keywords or []),
                "section_count": int(chapter_section_counts.get((bid, chno), 0)),
                "why": str(why3.get(f"{bid}:{chno}") or ""),
            })
        trace.append({"step": "Step3.raw", "raw": out3, "selected": step3_candidate_chapters, "available_chapters_total": {str(k): len(v) for k, v in allowed_chapters.items()}})
        _mark("Step3", t)

        # Step4 selected chapters (final chapters to inspect)
        t = time.perf_counter()
        executed_steps.append("Step4")
        _log("Step4", "确定最终查看的章")
        prompt4 = (
            f"根据下面的问题与候选章列表，选出最值得最终查看的 {1 if forced_retrieval else 0}-{topk_selected_chapters} 个章。\n"
            "输出 ONLY JSON: {\"selected\":[[book_id, chapter_no], ...], \"why\": {\"book_id:chapter_no\": \"理由\"}}。\n"
            f"只能从这些候选章中选择：{cand_pairs}.\n\n[问题]\n{q}\n\n[候选章]\n" + "\n\n".join([
                f"{x['chapter_label']}\nwhy={x.get('why','')}" for x in step3_candidate_chapters
            ])
        )
        out4 = call_json(session, rank4_model, system_msgs + [{"role": "user", "content": prompt4}], max_tokens=rank_max, temperature=0.0)
        step4_json_ok = isinstance(out4, dict) and isinstance(out4.get("selected"), list)
        sel_pairs = _coerce_pairs(out4.get("selected") if isinstance(out4, dict) else None, max_n=topk_selected_chapters, allowed_books=selected_books, allowed_chapters=allowed_chapters)
        # keep only from cand_pairs
        cand_set = set(cand_pairs)
        sel_pairs = [x for x in sel_pairs if x in cand_set]
        if forced_retrieval and not sel_pairs:
            sel_pairs = cand_pairs[:max(1, topk_selected_chapters)]
        elif not sel_pairs:
            sel_pairs = cand_pairs[: min(len(cand_pairs), topk_selected_chapters)]
        why4 = out4.get("why") if isinstance(out4, dict) and isinstance(out4.get("why"), dict) else {}
        for bid, chno in sel_pairs:
            cm = next((x for x in step3_candidate_chapters if x["book_id"] == bid and x["chapter_no"] == chno), None)
            if not cm:
                continue
            step4_selected_chapters.append({
                "book_id": bid,
                "chapter_no": chno,
                "chapter_title": cm["chapter_title"],
                "chapter_label": cm["chapter_label"],
                "chapter_summary": str(cm.get("chapter_summary") or ""),
                "chapter_keywords": list(cm.get("chapter_keywords") or []),
                "section_count": int(cm.get("section_count") or 0),
                "why": str(why4.get(f"{bid}:{chno}") or cm.get("why") or ""),
            })
        trace.append({"step": "Step4.raw", "raw": out4, "selected": step4_selected_chapters})
        _mark("Step4", t)

        # Step5 selected sections or chapter-only branch
        total_section_chars = 0
        step5_available_sections_total = 0
        if not chapter_only_mode:
            t = time.perf_counter()
            executed_steps.append("Step5")
            _log("Step5", "节级选择与读取")
            allowed_section_ids: Dict[Tuple[int, int], List[str]] = {}
            sec_lines: List[str] = []
            for ch_item in step4_selected_chapters:
                bid, chno = int(ch_item["book_id"]), int(ch_item["chapter_no"])
                secs = self.lib.list_sections(bid, chno, filter_policy={
                    "prefer_exposure": prefer_exposure,
                    "allow_masked_sections": allow_masked_sections,
                    "min_section_quality_score": min_section_quality,
                })
                allowed_section_ids[(bid, chno)] = [s.section_id for s in secs]
                step5_available_sections_total += len(secs)
                for s in secs:
                    sec_lines.append(
                        f"book_id={bid}\nchapter_no={chno}\nsection_id={s.section_id}\nsection_label={make_section_label(ch_item['chapter_label'], s.title)}\nsummary={s.summary}\nkeywords={', '.join(s.keywords)}\nexposure={s.exposure_decision}\nquality={s.quality_label}:{s.quality_score}"
                    )

            sel_triplets: List[Tuple[int, int, str]] = []
            why5: Dict[str, str] = {}
            if skip_section_ranking:
                # deterministic rule-based fallback: fill by chapter order and section order
                for ch_item in step4_selected_chapters:
                    pair = (int(ch_item["book_id"]), int(ch_item["chapter_no"]))
                    for sid in allowed_section_ids.get(pair, []):
                        sel_triplets.append((pair[0], pair[1], sid))
                        why5[f"{pair[0]}:{pair[1]}:{sid}"] = "rule_based_fill"
                        if len(sel_triplets) >= step5_sections_total_max:
                            break
                    if len(sel_triplets) >= step5_sections_total_max:
                        break
                step5_json_ok = True
                step5_selection_mode = "rule_based"
                out5 = {"selected": [[a, b, c] for a, b, c in sel_triplets], "why": why5}
            else:
                prompt5 = (
                    f"根据下面的问题与候选节信息，选出最相关的节，总数不超过 {step5_sections_total_max}。\n"
                    "输出 ONLY JSON: {\"selected\":[[book_id, chapter_no, section_id], ...], \"why\": {\"book_id:chapter_no:section_id\": \"理由\"}}。\n"
                    "只能从给定候选节中选择；不得重复。\n\n"
                    f"[问题]\n{q}\n\n[候选节]\n" + "\n\n".join(sec_lines)
                )
                out5 = call_json(session, rank5_model, system_msgs + [{"role": "user", "content": prompt5}], max_tokens=rank_max, temperature=0.0)
                step5_json_ok = isinstance(out5, dict) and isinstance(out5.get("selected"), list)
                sel_triplets = _coerce_triplets(out5.get("selected") if isinstance(out5, dict) else None, max_n=step5_sections_total_max, allowed_pairs=allowed_section_ids)
                why5 = out5.get("why") if isinstance(out5, dict) and isinstance(out5.get("why"), dict) else {}
                # deterministic backfill to satisfy budget as much as possible
                target_n = min(step5_sections_total_max, step5_available_sections_total)
                if len(sel_triplets) < target_n:
                    seen = set(sel_triplets)
                    for ch_item in step4_selected_chapters:
                        pair = (int(ch_item["book_id"]), int(ch_item["chapter_no"]))
                        for sid in allowed_section_ids.get(pair, []):
                            if len(sel_triplets) >= target_n:
                                break
                            item = (pair[0], pair[1], sid)
                            if item in seen:
                                continue
                            sel_triplets.append(item)
                            seen.add(item)
                        if len(sel_triplets) >= target_n:
                            break
                step5_selection_mode = "llm_ranked"

            required_sections = 1 if forced_retrieval else 0
            if mode0 == "M8":
                required_sections = min(step5_sections_total_max, step5_available_sections_total)
            if forced_retrieval and len(sel_triplets) < required_sections:
                raise ValueError(f"Section retrieval insufficient: selected={len(sel_triplets)} required={required_sections} available={step5_available_sections_total}")

            for bid, chno, sid in sel_triplets:
                ch_item = next((x for x in step4_selected_chapters if int(x["book_id"]) == bid and int(x["chapter_no"]) == chno), None)
                secs = self.lib.list_sections(bid, chno, filter_policy={
                    "prefer_exposure": prefer_exposure,
                    "allow_masked_sections": allow_masked_sections,
                    "min_section_quality_score": min_section_quality,
                })
                sm = next((s for s in secs if s.section_id == sid), None)
                if not sm:
                    continue
                section_label = make_section_label(ch_item["chapter_label"] if ch_item else make_chapter_label(self.lib.get_entry(bid).title if self.lib.get_entry(bid) else f"book{bid}", chno, ""), sm.title)
                text = _truncate_chars(self.lib.read_section_text(bid, chno, sid), section_text_max_chars)
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
            trace.append({"step": "Step5.raw", "raw": out5, "selected": step5_selected_sections, "available_sections_total": step5_available_sections_total, "selection_mode": step5_selection_mode})
            _mark("Step5", t)
        else:
            step5_selection_mode = "chapter_only"

        # Step6 answer
        t = time.perf_counter()
        executed_steps.append("Step6")
        if chapter_only_mode:
            _log("Step6", "基于最终查看的章级摘要回答问题")
            knowledge_blocks: List[str] = []
            for ch in step4_selected_chapters:
                chapter_summary_text = str(ch.get("chapter_summary") or "").strip() or str(ch.get("chapter_title") or "").strip()
                kw = ", ".join(list(ch.get("chapter_keywords") or []))
                knowledge_blocks.append(
                    f"[来源] {ch['chapter_label']}\n[章摘要]\n{chapter_summary_text}\n[关键词]\n{kw}\n理由：{ch.get('why','')}"
                )
                final_sources.append({"chapter_label": ch["chapter_label"], "why_used": ch.get("why", "")})
            prompt6 = (
                "根据下面的章级知识回答问题，并严格按如下格式输出：\n"
                "【解题思路与公式推导思路】\n- 说明使用了哪些章、哪些关键公式与推导步骤。\n"
                "【最终答案】\n- 直接给出结论/表达式/数值；不同小题分段。\n\n"
                f"[问题]\n{q}\n\n[章级知识]\n" + "\n\n".join(knowledge_blocks)
            )
        else:
            _log("Step6", "基于节级内容回答问题")
            knowledge_blocks = []
            for v in section_views:
                knowledge_blocks.append(f"[来源] {v['section_label']}\n[内容]\n{v['text_excerpt']}\n")
                final_sources.append({"section_label": v["section_label"], "why_used": v.get("why", "")})
            prompt6 = (
                "根据下面按节组织的知识回答问题，并严格按如下格式输出：\n"
                "【解题思路与公式推导思路】\n- 说明将使用哪些来源、哪些关键公式与推导步骤。\n"
                "【最终答案】\n- 直接给出结论/表达式/数值；不同小题分段。\n\n"
                f"[问题]\n{q}\n\n[知识]\n" + "\n\n".join(knowledge_blocks)
            )
        answer2_raw = call_chat(session, answer_model, system_msgs + [{"role": "user", "content": prompt6}], max_tokens=final_max, temperature=0.2).strip()
        answer2 = self._enforce_answer_format(
            session,
            stage_name="Step6",
            question=q,
            raw_answer=answer2_raw,
            generation_model=answer_model,
            generation_prompt=prompt6,
            repair_model=format_model,
            system_msgs=system_msgs,
            generation_max_tokens=final_max,
            repair_max_tokens=answer_format_repair_max,
            trace=trace,
        )
        trace.append({"step": "Step6.answer2", "answer2": answer2})
        _mark("Step6", t)

        t = time.perf_counter()
        cmp_prompt = (
            "比较下面同一问题的两个答案，优先级：正确性 > 完整性/无截断 > 表达清晰。"
            "若某答案有未闭合代码块、重复标题、明显截断尾巴，则判为更差。"
            "只返回 A 或 B。A=答案1更好，B=答案2更好。\n\n"
            f"[问题]\n{q}\n\n[答案1]\n{answer1}\n\n[答案2]\n{answer2}\n"
        )
        cmp_raw = call_chat(session, compare_model, system_msgs + [{"role": "user", "content": cmp_prompt}], max_tokens=compare_max, temperature=0.0)
        cmp = _parse_letter(cmp_raw, "AB")
        selected = "answer1" if cmp == "A" else "answer2"
        issues1 = _integrity_issues(answer1)
        issues2 = _integrity_issues(answer2)
        if selected == "answer1" and _is_severely_broken(issues1) and not _is_severely_broken(issues2):
            selected = "answer2"
            trace.append({"step": "SelectionOverride", "from": "answer1", "to": "answer2", "reason": issues1})
        if selected == "answer2" and _is_severely_broken(issues2) and not _is_severely_broken(issues1):
            selected = "answer1"
            trace.append({"step": "SelectionOverride", "from": "answer2", "to": "answer1", "reason": issues2})
        trace.append({"step": "Step6.compare", "raw": cmp_raw, "choice": cmp, "selected": selected, "answer1_issues": issues1, "answer2_issues": issues2})
        _mark("Compare", t)

        final_raw = answer1 if selected == "answer1" else answer2
        retrieval_context = {
            "step1_candidates_books": step1_candidates_books,
            "step2_selected_books": step2_selected_books,
            "step3_candidate_chapters": step3_candidate_chapters,
            "step4_selected_chapters": step4_selected_chapters,
            "step5_selected_sections": step5_selected_sections,
            "section_views": section_views,
            "final_sources": final_sources,
        }
        t = time.perf_counter()
        fmt = self._post_format_answer(session, q, final_raw, retrieval_context=retrieval_context, model=format_model, max_tokens=post_format_max)
        report_fields = self._build_structured_report_fields(question=q, raw_answer=final_raw, final_reasoning=fmt['final_reasoning'], retrieval_context=retrieval_context)
        final = f"【解题思路与公式推导思路】\n{fmt['final_reasoning']}\n\n【最终答案】\n{fmt['final_answer']}".strip()
        _mark("PostFormat", t)

        duration_sec = round(time.perf_counter() - t_all_start, 6)
        res = QAResult(
            question=q,
            answer=final,
            answer1=answer1,
            answer2=answer2,
            verdict0=verdict0,
            verdict0_confidence=verdict0_confidence,
            verdict0_rationale=verdict0_rationale,
            verdict0_missing_elements=verdict0_missing_elements,
            verdict0_explicit_errors=verdict0_explicit_errors,
            verdict0_uncertainties=verdict0_uncertainties,
            verdict0_gate_reason=(next((x.get("gate_reason", "") for x in reversed(trace) if x.get("step") == "Step0.verdict"), "")),
            verdict0_direct_sufficient=verdict0_direct_sufficient,
            verdict0_should_use_tools=verdict0_should_use_tools,
            verdict0_route_action=verdict0_route_action,
            verdict0_question_flags=verdict0_question_flags,
            verdict0_confidence_threshold_used=verdict0_confidence_threshold_used,
            selected=selected,
            outline=fmt["final_reasoning"],
            audit_outline=fmt["audit_reasoning"],
            final_answer=fmt["final_answer"],
            answer_raw=final_raw,
            report_question=report_fields["report_question"],
            report_retrieval_content=report_fields["report_retrieval_content"],
            report_key_steps=report_fields["report_key_steps"],
            report_solution_process=report_fields["report_solution_process"],
            trace=trace,
            step1_candidates_books=step1_candidates_books,
            step2_selected_books=step2_selected_books,
            step3_candidate_chapters=step3_candidate_chapters,
            step4_selected_chapters=step4_selected_chapters,
            step5_selected_sections=step5_selected_sections,
            section_views=section_views,
            final_sources=final_sources,
            run_dir=str(out_dir or ""),
            book_ids=[x["book_id"] for x in step1_candidates_books],
            book_chapters=[(x["book_id"], x["chapter_no"]) for x in step4_selected_chapters],
            extracts=legacy_extracts,
            test_mode=mode0,
            executed_steps=executed_steps,
            retrieval_used=True,
            retrieval_rounds=(len(step4_selected_chapters) if chapter_only_mode else len([x for x in section_views if str(x.get("text_excerpt") or "").strip()])),
            forced_retrieval=bool(forced_retrieval),
            no_retrieval=False,
            expert_prompt_used=expert_prompt_used,
            chapter_only_mode=chapter_only_mode,
            skip_section_ranking=skip_section_ranking,
            topk_books=topk_books,
            topk_books_view=topk_books_view,
            topk_candidate_chapters=topk_candidate_chapters,
            topk_selected_chapters=topk_selected_chapters,
            topk_sections=topk_sections,
            section_budget_used=len(step5_selected_sections),
            step5_selection_mode=step5_selection_mode,
            p_solve=p_solve,
            p_solve_reason=p_solve_reason,
            duration_sec=duration_sec,
            step_durations_sec=step_durations_sec,
            confidence_threshold=(confidence_threshold if mode0 == "M5" else None),
            m5_gate_triggered_retrieval=m5_gate_triggered_retrieval,
            section_chars_total=total_section_chars,
            step5_available_sections_total=step5_available_sections_total,
            step1_json_ok=step1_json_ok,
            step2_json_ok=step2_json_ok,
            step3_json_ok=step3_json_ok,
            step4_json_ok=step4_json_ok,
            step5_json_ok=step5_json_ok,
        )
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "qa_result.json").write_text(json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return res

    def _extract_key_steps_from_reasoning(self, final_reasoning: str, fallback_reasoning: str = "") -> str:
        text = str(final_reasoning or "").strip()
        if text:
            m = re.search(r"【\s*关键推导链条\s*】", text)
            if m:
                return text[m.end():].strip() or (fallback_reasoning or "（未提供）").strip()
        return (fallback_reasoning or text or "（未提供）").strip()

    def _build_report_retrieval_content(self, retrieval_context: Dict[str, Any]) -> str:
        max_items = int(self._qa_val("report_retrieval_max_items", 8) or 8)
        excerpt_max = int(self._qa_val("report_retrieval_excerpt_max_chars", 2500) or 2500)
        if not retrieval_context.get("step1_candidates_books"):
            return "本题未执行检索，最终答案来自直接作答路径。"

        lines: List[str] = []
        section_views = retrieval_context.get("section_views") or []
        if isinstance(section_views, list) and section_views:
            for idx, view in enumerate(section_views[:max_items], start=1):
                if not isinstance(view, dict):
                    continue
                label = str(view.get("section_label") or view.get("source_label") or view.get("section_title") or f"节{idx}").strip()
                why = str(view.get("why") or view.get("why_used") or "").strip()
                excerpt = str(view.get("text_excerpt") or "").strip()
                if excerpt_max > 0 and len(excerpt) > excerpt_max:
                    excerpt = excerpt[:excerpt_max].rstrip() + "\n...(truncated)..."
                lines.append(f"- 来源：{label}")
                lines.append(f"  选择理由：{why or '（未记录）'}")
                lines.append("  检索内容：")
                lines.append(_indent_block(excerpt or "（无可展示文本）", prefix="    "))
            if len(section_views) > max_items:
                lines.append(f"- 其余 {len(section_views) - max_items} 条检索内容已省略。")
            return "\n".join(lines).strip()

        chapters = retrieval_context.get("step4_selected_chapters") or []
        if isinstance(chapters, list) and chapters:
            for idx, ch in enumerate(chapters[:max_items], start=1):
                if not isinstance(ch, dict):
                    continue
                label = str(ch.get("chapter_label") or ch.get("source_label") or ch.get("chapter_title") or f"章{idx}").strip()
                why = str(ch.get("why") or ch.get("why_used") or "").strip()
                summary = str(ch.get("chapter_summary") or ch.get("chapter_title") or "").strip()
                keywords = [str(x).strip() for x in (ch.get("chapter_keywords") or []) if str(x).strip()]
                block_parts = []
                if summary:
                    block_parts.append("[章摘要]\n" + summary)
                if keywords:
                    block_parts.append("[关键词]\n" + ", ".join(keywords))
                block = "\n\n".join(block_parts).strip()
                if excerpt_max > 0 and len(block) > excerpt_max:
                    block = block[:excerpt_max].rstrip() + "\n...(truncated)..."
                lines.append(f"- 来源：{label}")
                lines.append(f"  选择理由：{why or '（未记录）'}")
                lines.append("  检索内容：")
                lines.append(_indent_block(block or "（无可展示文本）", prefix="    "))
            if len(chapters) > max_items:
                lines.append(f"- 其余 {len(chapters) - max_items} 条检索内容已省略。")
            return "\n".join(lines).strip()

        sources = retrieval_context.get("final_sources") or []
        if isinstance(sources, list) and sources:
            for idx, src in enumerate(sources[:max_items], start=1):
                if not isinstance(src, dict):
                    continue
                label = str(src.get("section_label") or src.get("chapter_label") or src.get("source_label") or f"来源{idx}").strip()
                why = str(src.get("why_used") or src.get("why") or "").strip()
                lines.append(f"- 来源：{label}")
                lines.append(f"  选择理由：{why or '（未记录）'}")
            if len(sources) > max_items:
                lines.append(f"- 其余 {len(sources) - max_items} 条检索内容已省略。")
            return "\n".join(lines).strip()

        return "本题执行了检索，但当前结果中未找到可展示的检索内容。"

    def _build_structured_report_fields(self, *, question: str, raw_answer: str, final_reasoning: str, retrieval_context: Dict[str, Any]) -> Dict[str, str]:
        solution_reasoning, _final = _split_answer_sections(raw_answer)
        solution_process = (solution_reasoning or raw_answer or "").strip() or "（未提供）"
        key_steps = self._extract_key_steps_from_reasoning(final_reasoning, solution_process)
        retrieval_content = self._build_report_retrieval_content(retrieval_context)
        return {
            "report_question": str(question or "").strip(),
            "report_retrieval_content": retrieval_content,
            "report_key_steps": key_steps,
            "report_solution_process": solution_process,
        }

    def _post_format_answer(self, session, question: str, raw_answer: str, *, retrieval_context: Dict[str, Any], model: str = "", max_tokens: int = 12000) -> Dict[str, str]:
        ctx_json = json.dumps(retrieval_context, ensure_ascii=False, indent=2)
        prompt = f"""你是一个严格的解题报告整理器。根据【问题】、【候选答案】和【检索选择链路(JSON)】，输出 ONLY JSON：
{{
  "final_reasoning": string,
  "final_answer": string,
  "audit_reasoning": string
}}

要求：
1. final_reasoning 仅包含两部分内容：
   - 【最终知识来源】
   - 【关键推导链条】
2. 【最终知识来源】中必须使用系统已给出的 source_label / section_label / chapter_label，不得杜撰来源。
3. audit_reasoning 必须包含：
   - 【查阅/检索轨迹】按步骤①-⑤记录：候选书、深入查看书、候选章、最终查看章、最终查看节；若是章级模式且未查看节，要明确写出“本模式停在章级”。
   - 【完整推导与计算】
4. final_answer 必须是简短直接的最终回答，不得复述 final_reasoning。
5. 所有数学符号/公式必须放在 $...$ 或 $$...$$ 中。
6. 不要输出任何额外文字，只输出 JSON。

[问题]
{question}

[候选答案]
{raw_answer}

[检索选择链路(JSON)]
{ctx_json}
"""
        out = call_json(session, model, [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.0)
        if not isinstance(out, dict) or out.get("_parse_error") or out.get("_exception"):
            reasoning, final = _split_answer_sections(raw_answer)
            fallback_audit = self._build_audit_fallback(question, reasoning or raw_answer, retrieval_context)
            fallback_reasoning = self._build_final_reasoning_fallback(reasoning or raw_answer, retrieval_context)
            return {"final_reasoning": _strip_heading_tags(fallback_reasoning), "final_answer": _strip_heading_tags(final), "audit_reasoning": _strip_heading_tags(fallback_audit)}
        fr = _strip_heading_tags(str(out.get("final_reasoning") or ""))
        fa = _strip_heading_tags(str(out.get("final_answer") or ""))
        ar = _strip_heading_tags(str(out.get("audit_reasoning") or ""))
        if not ar:
            ar = self._build_audit_fallback(question, fr or raw_answer, retrieval_context)
        if not fr:
            fr = self._build_final_reasoning_fallback(raw_answer, retrieval_context)
        if not fa:
            _r, fa = _split_answer_sections(raw_answer)
        return {"final_reasoning": fr, "final_answer": fa, "audit_reasoning": ar}

    def _build_final_reasoning_fallback(self, raw_reasoning: str, retrieval_context: Dict[str, Any]) -> str:
        lines: List[str] = ["【最终知识来源】"]
        sources = retrieval_context.get("final_sources") or []
        if sources:
            for s in sources:
                lines.append(f"- {s.get('section_label') or s.get('chapter_label') or s.get('source_label') or ''}：{s.get('why_used','')}")
        else:
            lines.append("- 此问题未查询书籍或章节。")
        lines.append("")
        lines.append("【关键推导链条】")
        lines.append(raw_reasoning.strip() or "（未提供）")
        return "\n".join(lines).strip()

    def _build_audit_fallback(self, question: str, raw_reasoning: str, retrieval_context: Dict[str, Any]) -> str:
        lines: List[str] = ["【查阅/检索轨迹】"]
        if not retrieval_context.get("step1_candidates_books"):
            lines.append("此问题未查询书籍或章节。")
            lines.append(f"全部推导基于题目所给信息：{question[:200]}；与模型已知的相关知识点：{raw_reasoning[:300]}。")
        else:
            lines.append("① 候选书（3-7本）")
            for x in retrieval_context.get("step1_candidates_books", []):
                lines.append(f"- {x.get('book_id')}｜{x.get('title')}｜理由：{x.get('rationale','')}")
            lines.append("② 深入查看书（1-2本）")
            for x in retrieval_context.get("step2_selected_books", []):
                lines.append(f"- {x.get('book_id')}｜{x.get('title')}｜理由：{x.get('why','')}")
            lines.append("③ 候选章")
            for x in retrieval_context.get("step3_candidate_chapters", []):
                lines.append(f"- {x.get('chapter_label')}")
            lines.append("④ 最终查看章")
            for x in retrieval_context.get("step4_selected_chapters", []):
                lines.append(f"- {x.get('chapter_label')}｜理由：{x.get('why','')}")
            lines.append("⑤ 最终查看节")
            if retrieval_context.get("step5_selected_sections"):
                final_set = {s.get('section_label') or s.get('source_label') for s in retrieval_context.get('final_sources', [])}
                for x in retrieval_context.get("step5_selected_sections", []):
                    lab = x.get('section_label') or x.get('source_label') or ''
                    used = "纳入最终思路" if lab in final_set else "未纳入最终思路"
                    lines.append(f"- {lab}｜理由：{x.get('why','')}｜{used}")
            else:
                lines.append("- 本模式停在章级，未进入节级查看。")
        lines.append("")
        lines.append("【完整推导与计算】")
        lines.append(raw_reasoning.strip() or "（未提供）")
        return "\n".join(lines).strip()
