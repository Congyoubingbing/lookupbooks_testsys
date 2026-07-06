from __future__ import annotations

import datetime
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ReportItem:
    qid: str
    title: str
    question: str
    retrieval_content: str = ""
    key_steps: str = ""
    solution_process: str = ""
    final_answer: str = ""
    score_text: str = ""
    outline: str = ""
    audit_outline: str = ""
    retrieval_chain: Dict[str, Any] = field(default_factory=dict)
    final_sources: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: str = ""


def _md_escape_fence(text: str) -> str:
    return (text or "").replace("```", "`\u200b``")


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "item"


def _render_list(lines: List[str], seq: List[str]) -> None:
    if not seq:
        lines.append("- （空）")
        return
    for s in seq:
        lines.append(f"- {s}")


def _extract_heading_content(text: str, heading: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    m = re.search(rf"【\s*{re.escape(heading)}\s*】", src)
    if not m:
        return ""
    rest = src[m.end():]
    n = re.search(r"\n【\s*[^\n]+?\s*】", rest)
    if n:
        return rest[:n.start()].strip()
    return rest.strip()


def _split_answer_sections(answer_text: str) -> tuple[str, str]:
    ans = (answer_text or "").strip()
    m1 = re.search(r"【\s*解题思路.*?】", ans)
    m2 = re.search(r"【\s*最终答案\s*】", ans)
    if m1 and m2 and m2.start() > m1.end():
        return ans[m1.end():m2.start()].strip(), ans[m2.end():].strip()
    return ans, ans


def _render_retrieval_chain(chain: Dict[str, Any], *, mode: str = "final") -> str:
    lines: List[str] = []
    lines.append("### 检索选择链路（自动记录）")
    lines.append("")
    step1 = chain.get("step1_candidates_books") or []
    step2 = chain.get("step2_selected_books") or []
    step3 = chain.get("step3_candidate_chapters") or []
    step4 = chain.get("step4_selected_chapters") or []
    step5 = chain.get("step5_selected_sections") or []
    if not any([step1, step2, step3, step4, step5]):
        lines.append("此问题未查询书籍或章节。")
        lines.append("")
        return "\n".join(lines)

    lines.append("#### ① 候选书（3-7本）")
    lines.append("")
    _render_list(lines, [f"{x.get('book_id')}｜{x.get('title')}｜理由：{x.get('rationale','')}" for x in step1])
    lines.append("")
    lines.append("#### ② 深入查看书（1-2本）")
    lines.append("")
    _render_list(lines, [f"{x.get('book_id')}｜{x.get('title')}｜理由：{x.get('why','')}" for x in step2])
    lines.append("")
    lines.append("#### ③ 候选章")
    lines.append("")
    _render_list(lines, [x.get("chapter_label", "") for x in step3])
    lines.append("")
    lines.append("#### ④ 最终查看章")
    lines.append("")
    _render_list(lines, [f"{x.get('chapter_label','')}｜理由：{x.get('why','')}" for x in step4])
    lines.append("")
    lines.append("#### ⑤ 最终查看节")
    lines.append("")
    _render_list(lines, [f"{x.get('section_label') or x.get('chapter_label') or x.get('source_label') or ''}｜理由：{x.get('why','')}" for x in step5])
    lines.append("")
    return "\n".join(lines)


def _render_final_sources(sources: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("### 最终知识来源（自动记录）")
    lines.append("")
    if not sources:
        lines.append("此问题未查询书籍或章节。")
        lines.append("")
        return "\n".join(lines)
    for x in sources:
        lines.append(f"- {x.get('section_label') or x.get('chapter_label') or x.get('source_label') or ''}｜采用原因：{x.get('why_used','')}")
    lines.append("")
    return "\n".join(lines)


def _fallback_retrieval_content(qa_result_payload: Dict[str, Any]) -> str:
    step1 = qa_result_payload.get("step1_candidates_books") or []
    if not step1:
        return "本题未执行检索，最终答案来自直接作答路径。"

    max_items = 8
    excerpt_max = 2500
    lines: List[str] = []

    section_views = qa_result_payload.get("section_views") or []
    if isinstance(section_views, list) and section_views:
        for view in section_views[:max_items]:
            if not isinstance(view, dict):
                continue
            label = str(view.get("section_label") or view.get("source_label") or view.get("section_title") or "").strip()
            why = str(view.get("why") or view.get("why_used") or "").strip() or "（未记录）"
            excerpt = str(view.get("text_excerpt") or "").strip()
            if excerpt_max > 0 and len(excerpt) > excerpt_max:
                excerpt = excerpt[:excerpt_max].rstrip() + "\n...(truncated)..."
            lines.append(f"- 来源：{label}")
            lines.append(f"  选择理由：{why}")
            lines.append("  检索内容：")
            lines.append("    " + (excerpt or "（无可展示文本）").replace("\n", "\n    "))
        if len(section_views) > max_items:
            lines.append(f"- 其余 {len(section_views) - max_items} 条检索内容已省略。")
        return "\n".join(lines).strip()

    step4 = qa_result_payload.get("step4_selected_chapters") or []
    if isinstance(step4, list) and step4:
        for ch in step4[:max_items]:
            if not isinstance(ch, dict):
                continue
            label = str(ch.get("chapter_label") or ch.get("source_label") or ch.get("chapter_title") or "").strip()
            why = str(ch.get("why") or ch.get("why_used") or "").strip() or "（未记录）"
            summary = str(ch.get("chapter_summary") or ch.get("chapter_title") or "").strip()
            keywords = [str(x).strip() for x in (ch.get("chapter_keywords") or []) if str(x).strip()]
            block_parts: List[str] = []
            if summary:
                block_parts.append("[章摘要]\n" + summary)
            if keywords:
                block_parts.append("[关键词]\n" + ", ".join(keywords))
            block = "\n\n".join(block_parts).strip()
            if excerpt_max > 0 and len(block) > excerpt_max:
                block = block[:excerpt_max].rstrip() + "\n...(truncated)..."
            lines.append(f"- 来源：{label}")
            lines.append(f"  选择理由：{why}")
            lines.append("  检索内容：")
            lines.append("    " + (block or "（无可展示文本）").replace("\n", "\n    "))
        if len(step4) > max_items:
            lines.append(f"- 其余 {len(step4) - max_items} 条检索内容已省略。")
        return "\n".join(lines).strip()

    final_sources = qa_result_payload.get("final_sources") or []
    if isinstance(final_sources, list) and final_sources:
        for src in final_sources[:max_items]:
            if not isinstance(src, dict):
                continue
            label = str(src.get("section_label") or src.get("chapter_label") or src.get("source_label") or "").strip()
            why = str(src.get("why_used") or src.get("why") or "").strip() or "（未记录）"
            lines.append(f"- 来源：{label}")
            lines.append(f"  选择理由：{why}")
        if len(final_sources) > max_items:
            lines.append(f"- 其余 {len(final_sources) - max_items} 条检索内容已省略。")
        return "\n".join(lines).strip()

    return "本题执行了检索，但当前结果中未找到可展示的检索内容。"


def _load_judge_payload(qid: str, q_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    if q_dir is None:
        return None
    candidates = [
        q_dir.parent / "_eval" / "judge_outputs" / f"{qid}.json",
        q_dir / "judge.json",
        q_dir / "judge_output.json",
    ]
    for p in candidates:
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            continue
    return None


def _format_score_text(judge_payload: Optional[Dict[str, Any]]) -> str:
    if not isinstance(judge_payload, dict):
        return "（当前尚无评分结果）"

    parsed = judge_payload.get("parsed") if isinstance(judge_payload.get("parsed"), dict) else judge_payload
    scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}
    if not isinstance(scores, dict) or not scores:
        flat = judge_payload
        maybe_keys = [
            "judge_score_overall",
            "judge_score_correctness",
            "judge_score_completeness",
            "judge_score_derivation",
            "judge_score_clarity",
            "judge_score_grounding",
            "judge_score_hallucination_resistance",
        ]
        if any(k in flat for k in maybe_keys):
            lines = [
                f"- overall: {flat.get('judge_score_overall', 'NA')}",
                f"- correctness: {flat.get('judge_score_correctness', 'NA')}",
                f"- completeness: {flat.get('judge_score_completeness', 'NA')}",
                f"- derivation: {flat.get('judge_score_derivation', 'NA')}",
                f"- clarity: {flat.get('judge_score_clarity', 'NA')}",
                f"- grounding: {flat.get('judge_score_grounding', 'NA')}",
                f"- hallucination_resistance: {flat.get('judge_score_hallucination_resistance', 'NA')}",
            ]
            return "\n".join(lines)
        return "（当前尚无评分结果）"

    def _fmt_list(name: str, value: Any) -> str:
        if not isinstance(value, list) or not value:
            return f"- {name}: []"
        return f"- {name}:\n" + "\n".join([f"  - {str(x).strip()}" for x in value if str(x).strip()])

    lines = [
        f"- overall: {scores.get('overall', 'NA')}",
        f"- correctness: {scores.get('correctness', 'NA')}",
        f"- completeness: {scores.get('completeness', 'NA')}",
        f"- derivation: {scores.get('derivation', 'NA')}",
        f"- clarity: {scores.get('clarity', 'NA')}",
        f"- grounding: {scores.get('grounding', 'NA')}",
        f"- hallucination_resistance: {scores.get('hallucination_resistance', 'NA')}",
        f"- should_use_tools: {parsed.get('should_use_tools', 'NA')}",
        f"- confidence: {parsed.get('confidence', 'NA')}",
        f"- used_evidence_indices: {parsed.get('used_evidence_indices', [])}",
        _fmt_list("key_issues", parsed.get("key_issues")),
        _fmt_list("unsupported_claims", parsed.get("unsupported_claims")),
        _fmt_list("strengths", parsed.get("strengths")),
    ]
    return "\n".join(lines).strip()


def render_markdown_report(
    items: Iterable[ReportItem],
    *,
    title: str,
    meta: Optional[Dict[str, Any]] = None,
    mode: str = "final",
) -> str:
    meta = meta or {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    items_list = list(items)
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- 生成时间：{ts}")
    for k, v in meta.items():
        lines.append(f"- {k}：{v}")
    lines.append("")
    lines.append("## 目录")
    lines.append("")
    for it in items_list:
        safe_title = it.title.strip() or "(无标题)"
        lines.append(f"- [{it.qid}｜{safe_title}](#{it.qid.lower()}-{_slugify(safe_title)})")
    lines.append("")

    for it in items_list:
        safe_title = it.title.strip() or "(无标题)"
        lines.append(f"## {it.qid}｜{safe_title}")
        lines.append("")
        if it.status != "ok":
            lines.append("### 状态")
            lines.append("")
            lines.append(f"**失败**：{it.error}")
            lines.append("")
        lines.append("### 题目")
        lines.append("")
        lines.append(_md_escape_fence(it.question).strip() or "（未提供）")
        lines.append("")
        lines.append("### 检索内容")
        lines.append("")
        lines.append((it.retrieval_content or "（未提供）").strip() or "（未提供）")
        lines.append("")
        lines.append("### 关键推导步骤")
        lines.append("")
        lines.append((it.key_steps or "（未提供）").strip() or "（未提供）")
        lines.append("")
        lines.append("### 实际的具体解题过程")
        lines.append("")
        lines.append((it.solution_process or "（未提供）").strip() or "（未提供）")
        lines.append("")
        lines.append("### 最终答案")
        lines.append("")
        lines.append((it.final_answer or "（未提供）").strip() or "（未提供）")
        lines.append("")
        lines.append("### 评分")
        lines.append("")
        lines.append((it.score_text or "（当前尚无评分结果）").strip() or "（当前尚无评分结果）")
        lines.append("")
        if mode == "audit":
            lines.append(_render_retrieval_chain(it.retrieval_chain, mode=mode))
            lines.append(_render_final_sources(it.final_sources))
            lines.append("### 审查版完整推导与检索轨迹")
            lines.append("")
            lines.append((it.audit_outline or "（未提供）").strip() or "（未提供）")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_markdown_index(
    items: Iterable[ReportItem],
    *,
    title: str,
    meta: Optional[Dict[str, Any]] = None,
    mode: str = "final",
    link_mode: str = "relative",
    base_dir: Optional[Path] = None,
) -> str:
    meta = meta or {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    items_list = list(items)

    def _link(qid: str) -> str:
        fname = "report_audit.md" if mode == "audit" else "report.md"
        if link_mode == "absolute" and base_dir is not None:
            return str((base_dir / qid / fname).resolve())
        return f"{qid}/{fname}"

    lines = [f"# {title}", "", f"- 生成时间：{ts}"]
    for k, v in meta.items():
        lines.append(f"- {k}：{v}")
    lines.append("")
    lines.append("## 目录")
    lines.append("")
    for it in items_list:
        safe_title = it.title.strip() or "(无标题)"
        lines.append(f"- [{it.qid}｜{safe_title}]({_link(it.qid)})")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(path: Path, md: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(md, encoding="utf-8")
    os.replace(str(tmp), str(path))


def build_item_from_qa_result(
    *,
    qid: str,
    title: str,
    question: str,
    qa_result_payload: Dict[str, Any],
    status: str = "ok",
    error: str = "",
    q_dir: Optional[Path] = None,
    judge_payload: Optional[Dict[str, Any]] = None,
) -> ReportItem:
    report_question = str(qa_result_payload.get("report_question") or question or qa_result_payload.get("question") or "").strip()
    retrieval_content = str(qa_result_payload.get("report_retrieval_content") or "").strip()
    if not retrieval_content:
        retrieval_content = _fallback_retrieval_content(qa_result_payload)

    key_steps = str(qa_result_payload.get("report_key_steps") or "").strip()
    outline = str(qa_result_payload.get("outline") or "").strip()
    if not key_steps:
        key_steps = _extract_heading_content(outline, "关键推导链条") or outline or "（未提供）"

    solution_process = str(qa_result_payload.get("report_solution_process") or "").strip()
    answer_raw = str(qa_result_payload.get("answer_raw") or "").strip()
    if not solution_process:
        solution_reasoning, _ = _split_answer_sections(answer_raw)
        solution_process = solution_reasoning or answer_raw or "（未提供）"

    if judge_payload is None:
        judge_payload = _load_judge_payload(qid, q_dir)
    score_text = str(qa_result_payload.get("report_score") or "").strip()
    if not score_text:
        score_text = _format_score_text(judge_payload)

    return ReportItem(
        qid=qid,
        title=title,
        question=report_question,
        retrieval_content=retrieval_content,
        key_steps=key_steps,
        solution_process=solution_process,
        final_answer=str(qa_result_payload.get("final_answer") or "").strip(),
        score_text=score_text,
        outline=outline,
        audit_outline=str(qa_result_payload.get("audit_outline") or "").strip(),
        retrieval_chain={
            "step1_candidates_books": qa_result_payload.get("step1_candidates_books") or [],
            "step2_selected_books": qa_result_payload.get("step2_selected_books") or [],
            "step3_candidate_chapters": qa_result_payload.get("step3_candidate_chapters") or [],
            "step4_selected_chapters": qa_result_payload.get("step4_selected_chapters") or [],
            "step5_selected_sections": qa_result_payload.get("step5_selected_sections") or [],
        },
        final_sources=qa_result_payload.get("final_sources") or [],
        status=status,
        error=error,
    )
