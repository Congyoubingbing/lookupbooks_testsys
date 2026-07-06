from __future__ import annotations
import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .utils import sha1_bytes


def _is_responses_model(model: str) -> bool:
    m = str(model or "").strip().lower()
    return m in {"gpt-5.4-pro", "gpt-5.4-high"}


def _responses_input_from_messages(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    instructions_parts: List[str] = []
    input_items: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "user")).lower()
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and block.get("text"):
                        text_parts.append(str(block.get("text")))
                    elif block.get("type") == "image_url":
                        url = None
                        iv = block.get("image_url")
                        if isinstance(iv, dict):
                            url = iv.get("url")
                        elif isinstance(iv, str):
                            url = iv
                        if url:
                            text_parts.append(f"[image_url] {url}")
                else:
                    text_parts.append(str(block))
            content_text = "\n".join(text_parts).strip()
        else:
            content_text = str(content)
        if role == "system":
            if content_text.strip():
                instructions_parts.append(content_text)
            continue
        mapped_role = "assistant" if role == "assistant" else "user"
        input_items.append({
            "role": mapped_role,
            "content": [{"type": "input_text", "text": content_text}],
        })
    return "\n\n".join(instructions_parts).strip(), input_items


def _responses_extract_text(resp: Any) -> str:
    try:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt
    except Exception:
        pass
    try:
        out = getattr(resp, "output", None) or []
        parts: List[str] = []
        for item in out:
            content = getattr(item, "content", None) or []
            for c in content:
                ctype = getattr(c, "type", None)
                if ctype in {"output_text", "text"}:
                    t = getattr(c, "text", None)
                    if t:
                        parts.append(str(t))
        if parts:
            return "\n".join(parts)
    except Exception:
        pass
    return ""


def _is_retryable_exception(e: Exception) -> bool:
    """Best-effort classification of retryable API errors.

    DashScope/OpenAI-compatible endpoints may raise different exception classes
    depending on openai version. We avoid tight coupling and rely on status codes
    and message patterns.
    """
    msg = str(e).lower()
    # Permanent / policy-like failures: do not retry
    if "data_inspection_failed" in msg:
        return False
    if "must provide a model" in msg or "provide a model parameter" in msg:
        return False
    if "invalid api key" in msg or "authentication" in msg:
        return False
    # Retryable patterns
    if "timeout" in msg or "timed out" in msg:
        return True
    if "rate limit" in msg or "too many requests" in msg:
        return True
    if "temporarily" in msg or "server" in msg or "bad gateway" in msg:
        return True
    # Status code when available
    code = getattr(e, "status_code", None)
    if code is None:
        # openai exceptions sometimes attach response.status_code
        resp = getattr(e, "response", None)
        code = getattr(resp, "status_code", None)
    if isinstance(code, int) and code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    return False

def _pil_to_data_url(img, fmt="PNG") -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")

def call_chat(session, model: str, messages: List[Dict[str, Any]], *,
              extra_body: Optional[Dict[str, Any]] = None,
              max_tokens: Optional[int] = None,
              temperature: float = 0.0) -> str:
    """Chat completion wrapper.

    Why max_tokens is Optional:
    - Many OpenAI-compatible endpoints allow omitting max_tokens (server default).
    - In this project we often want *no artificial low cap*; we instead use a generous
      default at call sites, and if the server rejects due to output/context limits,
      we decay max_tokens and retry.
    """
    # DashScope (OpenAI-compatible) requires an explicit model name.
    if not model or not isinstance(model, str) or not model.strip():
        raise ValueError(
            "Model name is empty. Please set config.models.llm_model (or config.models.reasoning_model)."
        )
    use_responses = _is_responses_model(model)
    if use_responses:
        instructions, input_items = _responses_input_from_messages(messages)
        kwargs = dict(model=model, input=input_items, temperature=temperature)
        if instructions:
            kwargs["instructions"] = instructions
        if max_tokens is not None:
            kwargs["max_output_tokens"] = int(max_tokens)
        if extra_body:
            kwargs.update(extra_body)
    else:
        kwargs = dict(model=model, messages=messages, temperature=temperature)
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        if extra_body:
            kwargs["extra_body"] = extra_body
    max_retries = int(getattr(session, "max_retries", 0) or 0)
    backoff_s = list(getattr(session, "backoff_s", None) or [2.0, 4.0, 8.0])

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            if use_responses:
                resp = session.client.responses.create(**kwargs)
                return _responses_extract_text(resp) or ""
            resp = session.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # If output/context limits are hit, decay max_tokens and retry.
            # This keeps the system robust across different providers/models.
            token_field = "max_output_tokens" if use_responses else "max_tokens"
            if token_field in kwargs and ("max_tokens" in msg or "maximum context length" in msg or "context length" in msg or "max_output_tokens" in msg):
                try:
                    kwargs[token_field] = max(256, int(int(kwargs[token_field]) * 0.7))
                except Exception:
                    pass

            if attempt >= max_retries or not _is_retryable_exception(e):
                raise
            # bounded backoff
            delay = backoff_s[min(attempt, len(backoff_s) - 1)] if backoff_s else 1.0
            time.sleep(float(delay))
    if last_err:
        raise last_err
    return ""


def call_json(session, model: str, messages: List[Dict[str, Any]], *,
              extra_body: Optional[Dict[str, Any]] = None,
              max_tokens: Optional[int] = None,
              temperature: float = 0.0) -> Dict[str, Any]:
    """Best-effort JSON extraction wrapper.

    If max_tokens is None, the request omits the parameter (server default).
    """
    try:
        txt = call_chat(session, model, messages, extra_body=extra_body, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        return {"_raw": "", "_exception": True, "_error": str(e)}
    # robust JSON extraction
    start = txt.find("{")
    end = txt.rfind("}")
    if start >= 0 and end > start:
        candidate = txt[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # fallback: try parse as json lines
    try:
        return json.loads(txt)
    except Exception:
        return {"_raw": txt, "_parse_error": True}


def vl_score_is_toc(session, img, *, model: str, enable_thinking: bool=False, temperature: float = 0.0) -> Dict[str, Any]:
    data_url = _pil_to_data_url(img)
    prompt = (
        "You are classifying whether this page is the BOOK-LEVEL (global) Table of Contents for the whole book. "
        "Return ONLY valid JSON with keys: is_toc (bool), score (0..1), signals (list of short strings). "
        "A global TOC typically lists MANY different chapters/parts (e.g., Chapter 1,2,3..., Part I/II, Appendix, Index) "
        "with aligned page numbers and dot leaders. "
        "NEGATIVE examples (should be is_toc=false): (a) a single-chapter outline listing sections like 1.1, 1.2, 1.9; "
        "(b) list of figures/tables; (c) publisher/series/title pages (ISBN, Applied Mathematical Sciences, etc.). "
        "If most numbered entries share the same top-level prefix (e.g., all start with '1.'), it is NOT a global TOC."
    )
    messages = [
        {"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image_url","image_url":{"url":data_url}}
        ]}
    ]
    extra = {"enable_thinking": enable_thinking} if enable_thinking else None
    out = call_json(session, model, messages, extra_body=extra, max_tokens=256, temperature=float(temperature or 0.0))
    if not isinstance(out, dict):
        return {}
    if "is_true_start" not in out and "is_start" in out:
        out["is_true_start"] = bool(out.get("is_start"))
    if "is_start" not in out and "is_true_start" in out:
        out["is_start"] = bool(out.get("is_true_start"))
    if "heading_text" not in out and out.get("extracted_title") is not None:
        out["heading_text"] = out.get("extracted_title")
    if "extracted_title" not in out and out.get("heading_text") is not None:
        out["extracted_title"] = out.get("heading_text")
    return out

def vl_extract_toc_markdown(session, images: Any, *, model: str, enable_thinking: bool=False, temperature: float = 0.0) -> str:
    """Extract TOC as markdown from 1..N page images.

    Backwards compatible: callers may pass a single PIL image.
    """
    if images is None:
        images = []
    elif not isinstance(images, (list, tuple)):
        images = [images]
    parts = [{"type":"text","text":(
        "You are extracting a book Table of Contents from scanned images.\n"
        "Task: transcribe the TOC entries as markdown lines. Preserve chapter/section numbering and titles; keep page numbers.\n"
        "Rules:\n"
        "- Do NOT invent missing entries.\n"
        "- If OCR is uncertain, keep best guess but do not hallucinate new lines.\n"
        "- Output ONLY markdown text.\n"
    )}]
    # Keep the batch small; large batches tend to collapse into page-number-only outputs.
    for img in list(images)[:6]:
        parts.append({"type":"image_url","image_url":{"url":_pil_to_data_url(img)}})
    messages = [{"role":"user","content":parts}]
    extra = {"enable_thinking": enable_thinking} if enable_thinking else None
    try:
        return call_chat(session, model, messages, extra_body=extra, max_tokens=4096, temperature=float(temperature or 0.0))
    except Exception as e:
        # Do not crash the pipeline on a single VL extraction failure; caller will mark fallback.
        return ""

def vl_read_page_label(
    session,
    crops: List[Any],
    *,
    model: str,
    enable_thinking: bool=False,
    crop_policy: Optional[str]=None,
    temperature: float = 0.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Provide multiple header/footer crops; ask VL to decide the page label (roman/arabic) if present.
    Returns JSON: {label: str|null, conf: 0..1, crop_index: int, notes: str}
    """
    # Accept either a single PIL image or an iterable of crops.
    # Older callers may pass a single image; wrap it to keep interface stable.
    if crops is None:
        crops = []
    elif not isinstance(crops, (list, tuple)):
        crops = [crops]

    parts = [{"type":"text","text":(
        "You are reading printed page number labels from scanned HEADER/FOOTER crops.\n"
        "Choose the crop that most clearly contains the printed page number.\n"
        "Return ONLY JSON: {label: string|null, conf: 0..1, crop_index: int|null, notes: string}.\n"
        "label should be exactly what is printed (e.g., 'vi', '27', '103'). If none, label=null.\n"
    )}]
    for img in crops:
        parts.append({"type":"image_url","image_url":{"url":_pil_to_data_url(img)}})
    messages = [{"role":"user","content":parts}]
    extra = {"enable_thinking": enable_thinking} if enable_thinking else None
    return call_json(session, model, messages, extra_body=extra, max_tokens=256, temperature=float(temperature or 0.0))

def vl_verify_chapter_start(session, img, chapter_no: int, chapter_title: str, *,
                            model: str, enable_thinking: bool=False, temperature: float = 0.0,
                            cfg=None) -> Dict[str, Any]:
    data_url = _pil_to_data_url(img)
    prompt = (
        "You are verifying whether this scanned page is the START of a specific book chapter (chapter-level only).\n"
        f"Target chapter number: {chapter_no}\n"
        f"Target chapter title (approx): {chapter_title}\n"
        "Return ONLY JSON: {is_true_start: bool, score: 0..1, heading_text: string|null, signals: list}.\n"
        "Rules: accept only chapter-level starts (not section/subsection starts like 4.1 or 7.1).\n"
        "Examples usually chapter-level: BROWNIAN MOTION, 第七章..., Chapter 7...\n"
        "Examples usually NOT chapter-level: 27. POLYMER MIXTURES, 4.1 Introduction.\n"
        "Signals may include: CHAPTER, large chapter number, large title, first-page layout.\n"
    )
    messages = [{"role":"user","content":[
        {"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":data_url}}
    ]}]
    extra = {"enable_thinking": enable_thinking} if enable_thinking else None
    max_tokens = 256
    try:
        if cfg is not None:
            max_tokens = int(getattr(getattr(cfg, "llm", None), "vl_boundary_verify_max_tokens", max_tokens) or max_tokens)
    except Exception:
        max_tokens = 256
    out = call_json(session, model, messages, extra_body=extra, max_tokens=max_tokens, temperature=float(temperature or 0.0))
    if not isinstance(out, dict):
        return {}
    if "is_true_start" not in out and "is_start" in out:
        out["is_true_start"] = bool(out.get("is_start"))
    if "is_start" not in out and "is_true_start" in out:
        out["is_start"] = bool(out.get("is_true_start"))
    if "heading_text" not in out and out.get("extracted_title") is not None:
        out["heading_text"] = out.get("extracted_title")
    if "extracted_title" not in out and out.get("heading_text") is not None:
        out["extracted_title"] = out.get("heading_text")
    return out

def vl_extract_opening_anchors(session, img, chapter_no: int, *, model: str, enable_thinking: bool=False,
                               temperature: float = 0.0,
                               snippet_words: int=220, anchors_per_chapter: int=3) -> Dict[str, Any]:
    data_url = _pil_to_data_url(img)
    prompt = (
        "You are extracting robust textual anchors from the opening page of a book chapter (scanned image).\n"
        f"Chapter number (if visible): {chapter_no}\n"
        "Return ONLY JSON with keys:\n"
        "- title: string|null (the chapter title as printed)\n"
        f"- anchors: list of up to {anchors_per_chapter} short anchor strings (prefer title + 2 short sentences from first paragraph)\n"
        f"- opening_snippet: a short plain-text snippet of ~{snippet_words} words from the start of the chapter (exclude formulas if possible)\n"
        "- conf: 0..1\n"
        "Rules:\n"
        "- Avoid formulas; prefer narrative sentences that likely appear in the TeX/TXT.\n"
        "- Do NOT invent content not on the page.\n"
    )
    messages = [{"role":"user","content":[
        {"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":data_url}}
    ]}]
    extra = {"enable_thinking": enable_thinking} if enable_thinking else None
    return call_json(session, model, messages, extra_body=extra, max_tokens=768, temperature=float(temperature or 0.0))


def vl_extract_short_phrases(session, img, *, model: str, enable_thinking: bool=False,
                             temperature: float = 0.0,
                             max_phrases: int=6) -> Dict[str, Any]:
    """Extract a few short, literal phrases from a scanned page image for text-side verification.

    Returns JSON: {phrases: [string], conf: 0..1}
    """
    data_url = _pil_to_data_url(img)
    prompt = (
        "Extract a few short phrases that literally appear on this page (body text preferred).\n"
        f"Return ONLY JSON: {{\"phrases\": [..], \"conf\": 0..1}}.\n"
        f"Rules:\n"
        f"- Provide up to {max_phrases} phrases.\n"
        "- Each phrase should be 5-15 words if possible (or 30-120 characters).\n"
        "- Avoid formulas, headers/footers, page numbers, and figure captions unless unavoidable.\n"
        "- Do NOT paraphrase; copy verbatim text fragments from the page.\n"
    )
    messages = [{"role":"user","content":[
        {"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":data_url}}
    ]}]
    extra = {"enable_thinking": enable_thinking} if enable_thinking else None
    return call_json(session, model, messages, extra_body=extra, max_tokens=512, temperature=float(temperature or 0.0))

def llm_parse_toc(
    session,
    toc_markdown: str,
    *,
    model: str,
    parse_depth: str = "chapter_only",
    max_chapters: int = 150,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    **_kwargs,
) -> Dict[str, Any]:
    """Parse TOC markdown into a strict JSON chapter list.

    NOTE: Some callers may pass enable_thinking (for DashScope-compatible "thinking" mode).
    Older versions of this function did not accept this kwarg; we accept it for backward
    compatibility and route it via extra_body.
    """
    depth = (parse_depth or "chapter_only").strip().lower()
    if depth not in {"chapter_only", "chapter_section", "entries", "auto"}:
        depth = "chapter_only"

    # Instruction variants
    if depth in {"entries", "auto"}:
        include_rule = (
            "- Include top-level TOC entries even if they are not labeled as 'Chapter'.\n"
            "- Accept entries like '1 Introduction .... 3', 'Part I ... 12', or plain titles with page numbers.\n"
            "- For 'no': use the explicit number if present; otherwise assign sequential integers starting at 1.\n"
            "- Recognize Chinese markers: '第1章/第一章/第十章', '附录', and (when chapters are absent) '第1篇'.\n"
        )
    elif depth == "chapter_section":
        include_rule = (
            "- Prefer Chapter-level entries when present, but also accept Part/Section-level entries if the TOC does not clearly mark chapters.\n"
            "- For 'no': use explicit chapter numbers when present; otherwise assign sequential integers starting at 1.\n"
            "- Recognize Chinese markers: '第1章/第一章/第十章', '附录', and '第1篇' (only if no clearer chapters exist).\n"
        )
    else:  # chapter_only
        include_rule = (
            "- Include only Chapter-level entries (e.g., 'Chapter 1', '第1章', '第一章', '附录').\n"
        )

    prompt = (
        "You are given a Table of Contents in markdown text extracted from a book.\n"
        "Task: parse it into strict JSON.\n"
        "Output schema:\n"
        "{\n"
        "  \"chapters\": [\n"
        "    {\"no\": int, \"title\": string, \"printed_page\": string}\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        + include_rule +
        "- 'printed_page' must be the printed page label as a STRING (roman like 'vi' or arabic like '97').\n"
        "- Do not invent entries not present in the TOC.\n"
        f"- Hard limit: at most {max_chapters} entries.\n"
        "Return ONLY JSON.\n"
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt + "\n\nTOC:\n" + toc_markdown}],
        }
    ]
    extra = {"enable_thinking": True} if enable_thinking else None
    return call_json(session, model, messages, extra_body=extra, max_tokens=2048, temperature=float(temperature or 0.0))