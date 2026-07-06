# -*- coding: utf-8 -*-
"""Suite runner for lookupbooks_testsys.

Goal
----
Run an end-to-end "pass" (smoke) test suite across M1–M8 with a single command,
without changing the default behavior of `python run.py batch-ask`.

Implementation notes
--------------------
We intentionally execute each stage as a subprocess invoking the same `run.py`
entrypoints (batch-ask / eval-batch / eval-compare). This guarantees that the
stage semantics are identical to manual runs and avoids invasive refactors.

Suite artifacts are written under:
  runs/suite/<suite_id>/<suite_run_id>/
and reference per-stage batch outputs under runs/batch/<batch_id>/.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from subprocess import CompletedProcess, Popen, PIPE, run
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from pdf_txt_align.utils import ensure_dir, dump_json, now_ts


def _tail(s: str, n: int = 4000) -> str:
    if not s:
        return ""
    s = str(s)
    return s[-n:]


def _parse_csv_ints(s: str, *, default: List[int]) -> List[int]:
    if not s:
        return list(default)
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    out = [x for x in out if x > 0]
    # keep order but unique
    seen = set()
    uniq: List[int] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq or list(default)


def _expand_suite_modes(spec: str) -> List[str]:
    """Return a list of modes requested by user.

    Accepted forms:
      - "M1-M9" / "all"
      - "M2,M3,M5"
      - "M8" (will run M8 k-sweep)
    """
    s = (spec or "").strip().upper()
    if not s:
        return []
    if s in {"ALL", "M1-M8", "M1–M8", "M1~M8", "M1-M9", "M1–M9", "M1~M9"}:
        return ["M1","M2","M3","M4","M5","M6","M7","M8","M9"]
    parts = [p.strip().upper() for p in s.replace(";", ",").split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        if p in {"M1-M8", "M1–M8", "M1-M9", "M1–M9"}:
            out.extend(["M1","M2","M3","M4","M5","M6","M7","M8","M9"])
        elif p.startswith("M") and p[1:].isdigit():
            out.append(p)
    # unique, keep order
    seen = set()
    uniq: List[str] = []
    for m in out:
        if m in seen:
            continue
        seen.add(m)
        uniq.append(m)
    return uniq


def _suite_order_filter(requested: List[str]) -> List[str]:
    # Recommended pass-test order (fast → heavy → ablations/sweep)
    order = ["M2", "M4", "M3", "M5", "M1", "M9", "M6", "M7", "M8"]
    if not requested:
        return []
    req = set(requested)
    return [m for m in order if m in req]


@dataclass
class StageResult:
    stage_id: str
    stage_name: str
    mode: str
    k: Optional[int] = None
    status: str = "pending"  # ok/failed/skipped
    started_at: str = ""
    finished_at: str = ""
    batch_id: str = ""
    batch_dir: str = ""
    eval_ok: bool = False
    eval_dir: str = ""
    compare_dir: str = ""
    command_batch_ask: List[str] = None  # type: ignore
    command_eval_batch: List[str] = None  # type: ignore
    returncode_batch_ask: Optional[int] = None
    returncode_eval_batch: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str = ""
    traceback: str = ""
    metrics: Dict[str, Any] = None  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # dataclasses default None lists need normalization
        if d.get("command_batch_ask") is None:
            d["command_batch_ask"] = []
        if d.get("command_eval_batch") is None:
            d["command_eval_batch"] = []
        if d.get("metrics") is None:
            d["metrics"] = {}
        return d


class _TailBuf:
    """Keep a rolling tail of text (by chars) to avoid unbounded memory."""

    def __init__(self, max_chars: int = 12000):
        self.max_chars = int(max_chars)
        self._q = deque()  # type: deque[str]
        self._n = 0
        self._lock = threading.Lock()

    def append(self, s: str) -> None:
        if not s:
            return
        with self._lock:
            self._q.append(s)
            self._n += len(s)
            while self._n > self.max_chars and self._q:
                x = self._q.popleft()
                self._n -= len(x)

    def get(self) -> str:
        with self._lock:
            return "".join(list(self._q))


def _run_subprocess(cmd: List[str], *, cwd: Path, echo: bool = True) -> CompletedProcess:
    """Run a subprocess.

    In suite mode we want *visible* progress in the console. Therefore we
    tee stdout/stderr to the parent process while also capturing a tail for
    the suite report.
    """

    if not echo:
        return run(cmd, cwd=str(cwd), capture_output=True, text=True)

    proc = Popen(cmd, cwd=str(cwd), stdout=PIPE, stderr=PIPE, text=True, bufsize=1)
    out_buf = _TailBuf(max_chars=20000)
    err_buf = _TailBuf(max_chars=20000)

    def _pump(stream, writer, buf: _TailBuf):
        try:
            for line in iter(stream.readline, ""):
                writer.write(line)
                writer.flush()
                buf.append(line)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t1 = threading.Thread(target=_pump, args=(proc.stdout, sys.stdout, out_buf), daemon=True)  # type: ignore
    t2 = threading.Thread(target=_pump, args=(proc.stderr, sys.stderr, err_buf), daemon=True)  # type: ignore
    t1.start(); t2.start()
    rc = proc.wait()
    t1.join(timeout=1.0); t2.join(timeout=1.0)

    return CompletedProcess(cmd, rc, stdout=out_buf.get(), stderr=err_buf.get())


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_stage_metrics(batch_dir: Path) -> Dict[str, Any]:
    """Collect lightweight per-stage metrics for suite report."""
    out: Dict[str, Any] = {}
    bs = _read_json(batch_dir / "batch_summary.json") or {}
    out["ok_count"] = bs.get("ok_count")
    out["fail_count"] = bs.get("fail_count")
    out["items_total"] = bs.get("items_total")
    es = _read_json(batch_dir / "_eval" / "summary.json")
    if isinstance(es, dict):
        out["eval_has_judge"] = bool(es.get("has_judge"))
        modes = es.get("modes") or {}
        # batch-ask in a stage runs a single mode; but we still store by_mode if present
        out["eval_modes"] = modes
    return out


def _write_suite_report_md(suite_root: Path, suite_id: str, suite_run_id: str, stages: List[StageResult]) -> Tuple[Path, Path]:
    # Final (compact)
    lines: List[str] = []
    lines.append(f"# lookupbooks_testsys suite report")
    lines.append("")
    lines.append(f"- suite_id: `{suite_id}`")
    lines.append(f"- suite_run_id: `{suite_run_id}`")
    lines.append(f"- generated_at: `{datetime.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    hdr = ["stage", "mode", "k", "status", "batch_id", "batch_dir", "eval_ok", "spec_ok_rate", "retrieval_rate", "error"]
    lines.append("|" + "|".join(hdr) + "|")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for st in stages:
        spec_ok_rate = ""
        retrieval_rate = ""
        try:
            modes = (st.metrics or {}).get("eval_modes") or {}
            md = modes.get(st.mode) or modes.get(str(st.mode or ""))
            if isinstance(md, dict):
                spec_ok_rate = f"{float(md.get('spec_ok_rate', 0.0)):.3f}"
                retrieval_rate = f"{float(md.get('retrieval_rate', 0.0)):.3f}"
        except Exception:
            pass
        err_short = (st.error or "").splitlines()[0][:80] if st.error else ""
        lines.append("|" + "|".join([
            st.stage_id,
            st.mode,
            str(st.k or ""),
            st.status,
            st.batch_id,
            st.batch_dir,
            "Y" if st.eval_ok else "N",
            spec_ok_rate,
            retrieval_rate,
            err_short.replace("|", "/"),
        ]) + "|")

    lines.append("")
    lines.append("## Notes")
    lines.append("- Per-stage detailed logs are in `stages/*.json`.")
    lines.append("- Per-stage batch outputs live under `runs/batch/<batch_id>/`.")
    lines.append("- Each batch's evaluation artifacts are under `runs/batch/<batch_id>/_eval/`.")
    lines.append("")

    out_md = suite_root / "suite_report.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    # Audit (verbose)
    audit: List[str] = []
    audit.append(f"# lookupbooks_testsys suite report (audit)")
    audit.append("")
    audit.append(f"- suite_id: `{suite_id}`")
    audit.append(f"- suite_run_id: `{suite_run_id}`")
    audit.append("")
    for st in stages:
        audit.append(f"## {st.stage_id} {st.stage_name}")
        audit.append("")
        audit.append(f"- mode: `{st.mode}`")
        if st.k is not None:
            audit.append(f"- k: `{st.k}`")
        audit.append(f"- status: `{st.status}`")
        audit.append(f"- batch_id: `{st.batch_id}`")
        audit.append(f"- batch_dir: `{st.batch_dir}`")
        audit.append(f"- eval_ok: `{st.eval_ok}`")
        audit.append("")
        if st.error:
            audit.append("### error")
            audit.append("```text")
            audit.append(st.error)
            audit.append("```")
        if st.traceback:
            audit.append("### traceback")
            audit.append("```text")
            audit.append(_tail(st.traceback, 12000))
            audit.append("```")
        if st.stderr_tail:
            audit.append("### stderr (tail)")
            audit.append("```text")
            audit.append(_tail(st.stderr_tail, 12000))
            audit.append("```")
        if st.stdout_tail:
            audit.append("### stdout (tail)")
            audit.append("```text")
            audit.append(_tail(st.stdout_tail, 12000))
            audit.append("```")
        audit.append("")

    out_audit = suite_root / "suite_report_audit.md"
    out_audit.write_text("\n".join(audit), encoding="utf-8")

    return out_md, out_audit


def run_batch_suite(args, *, run_py_path: Path) -> None:
    """Entry for `run.py batch-ask --suite ...`"""
    project_root = run_py_path.parent
    suite_id = str(getattr(args, "suite_id", "") or "smoke_testsys_small").strip() or "smoke_testsys_small"
    suite_spec = str(getattr(args, "suite", "") or "").strip()
    modes_req = _expand_suite_modes(suite_spec)
    modes = _suite_order_filter(modes_req)
    if not modes:
        raise SystemExit(f"Invalid --suite '{suite_spec}'. Use 'M1-M9' or 'M2,M3,...'.")

    ks = _parse_csv_ints(str(getattr(args, "suite_m8_ks", "") or ""), default=[1, 3, 5])
    suite_runs_root = ensure_dir(Path(str(getattr(args, "suite_runs_dir", "./runs/suite") or "./runs/suite")))
    suite_run_id = now_ts()
    suite_root = ensure_dir(suite_runs_root / suite_id / suite_run_id)
    stages_dir = ensure_dir(suite_root / "stages")

    # Preflight checks (no LLM calls)
    pre = StageResult(stage_id="01", stage_name="preflight", mode="PRE")
    pre.started_at = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        batch_file = Path(args.batch_file)
        agent_cfg = Path(args.agent_config)
        lib_root = Path(args.library_root)
        section_index_files = list((lib_root / "books").glob("book*/sections/ch*/section_index.json")) if (lib_root / "books").exists() else []
        section_text_files = list((lib_root / "books").glob("book*/sections/ch*/*.txt")) if (lib_root / "books").exists() else []
        chapter_summary_files = list((lib_root / "books").glob("book*_chapter_summary.txt")) if (lib_root / "books").exists() else []
        pre.metrics = {
            "batch_file_exists": batch_file.exists(),
            "agent_config_exists": agent_cfg.exists(),
            "outputs_root": str(Path(args.outputs_root).resolve()),
            "library_root": str(lib_root.resolve()),
            "runs_dir": str(Path(args.runs_dir).resolve()),
            # Your deployment uses 0-based env vars: QWEN_API_KEY_0 .. QWEN_API_KEY_9
            "api_keys_found": sum(1 for i in range(0, 10) if os.environ.get(f"QWEN_API_KEY_{i}")),
            "library_has_summarybook": (lib_root / "summarybook.txt").exists(),
            "library_books_dir_exists": (lib_root / "books").exists(),
            "library_chapter_summary_files": len(chapter_summary_files),
            "library_section_index_files": len(section_index_files),
            "library_section_text_files": len(section_text_files),
            "library_sections_ready": bool(section_index_files and section_text_files),
        }
        pre_ok = bool(pre.metrics.get("batch_file_exists")) and bool(pre.metrics.get("agent_config_exists")) and bool(pre.metrics.get("library_has_summarybook")) and bool(pre.metrics.get("library_sections_ready"))
        pre.status = "ok" if pre_ok else "failed"
        if pre.status != "ok":
            pre.error = "preflight failed: batch/config missing or library section-level artifacts not ready"
    except Exception as e:
        pre.status = "failed"
        pre.error = str(e)
        pre.traceback = traceback.format_exc()
    pre.finished_at = datetime.datetime.now().isoformat(timespec="seconds")
    dump_json(stages_dir / "01_preflight.json", pre.to_dict())

    stage_results: List[StageResult] = [pre]

    # Build stage list
    stage_specs: List[Tuple[str, str, Optional[int]]] = []  # (stage_id, mode, k)
    sid = 2
    for m in modes:
        if m == "M8":
            for k in ks:
                stage_specs.append((f"{sid:02d}", "M8", k))
                sid += 1
        else:
            stage_specs.append((f"{sid:02d}", m, None))
            sid += 1

    stop_on_error = bool(getattr(args, "suite_stop_on_error", False))
    enable_judge = bool(getattr(args, "suite_judge", False))

    # Determine rubric and judge workers (passed through to eval-batch)
    rubric_override = str(getattr(args, "suite_rubric", "") or "").strip()
    judge_workers_override = int(getattr(args, "suite_judge_workers", 0) or 0)
    skip_existing_judge = bool(getattr(args, "suite_skip_existing_judge", False))

    # Run stages
    successful_batch_dirs: List[Path] = []
    for stage_id, mode, k in stage_specs:
        name = mode if k is None else f"{mode}_k{k}"
        st = StageResult(stage_id=stage_id, stage_name=name, mode=mode, k=k)
        st.started_at = datetime.datetime.now().isoformat(timespec="seconds")
        st.status = "pending"

        batch_id = f"{suite_id}_{suite_run_id}_{name}"
        st.batch_id = batch_id
        batch_dir = Path(args.runs_dir) / batch_id
        st.batch_dir = str(batch_dir.resolve())

        # Build batch-ask command (no --suite to avoid recursion)
        cmd_ba: List[str] = [
            sys.executable,
            "-u",
            str(run_py_path),
            "batch-ask",
            "--batch-file", str(args.batch_file),
            "--tag", str(args.tag),
            "--agent-config", str(args.agent_config),
            "--outputs-root", str(args.outputs_root),
            "--library-root", str(args.library_root),
            "--runs-dir", str(args.runs_dir),
            "--batch-id", batch_id,
            "--start", str(int(args.start)),
            "--end", str(int(args.end)),
            "--test-mode", mode,
        ]
        if args.workers is not None:
            cmd_ba += ["--workers", str(int(args.workers))]
        if getattr(args, "auto_summarize", False):
            cmd_ba.append("--auto-summarize")
        if getattr(args, "verbose", False):
            cmd_ba.append("--verbose")
        if getattr(args, "stop_on_error", False):
            cmd_ba.append("--stop-on-error")
        if k is not None:
            cmd_ba += ["--topk-books", str(k), "--topk-sections", str(k)]

        st.command_batch_ask = cmd_ba

        try:
            print(f"\n[SUITE] === Stage {stage_id} {name} ===")
            print("[SUITE] batch-ask:", " ".join(cmd_ba))
            proc = _run_subprocess(cmd_ba, cwd=project_root)
            st.returncode_batch_ask = proc.returncode
            st.stdout_tail = _tail(proc.stdout)
            st.stderr_tail = _tail(proc.stderr)
            if proc.returncode != 0:
                st.status = "failed"
                st.error = f"batch-ask failed with returncode={proc.returncode}"
            else:
                st.status = "ok"
        except Exception as e:
            st.status = "failed"
            st.error = str(e)
            st.traceback = traceback.format_exc()

        # If batch failed, record and (maybe) continue
        if st.status != "ok":
            st.finished_at = datetime.datetime.now().isoformat(timespec="seconds")
            dump_json(stages_dir / f"{stage_id}_{name}.json", st.to_dict())
            stage_results.append(st)
            if stop_on_error:
                break
            continue

        # Eval-batch (auto metrics always; judge optional)
        cmd_ev: List[str] = [
            sys.executable,
            "-u",
            str(run_py_path),
            "eval-batch",
            "--batch-dir", str(batch_dir),
            "--agent-config", str(args.agent_config),
        ]
        if enable_judge:
            cmd_ev.append("--judge")
            if rubric_override:
                cmd_ev += ["--rubric", rubric_override]
            if judge_workers_override > 0:
                cmd_ev += ["--judge-workers", str(judge_workers_override)]
            if skip_existing_judge:
                cmd_ev.append("--skip-existing")

        st.command_eval_batch = cmd_ev
        try:
            print("[SUITE] eval-batch:", " ".join(cmd_ev))
            proc2 = _run_subprocess(cmd_ev, cwd=project_root)
            st.returncode_eval_batch = proc2.returncode
            st.stdout_tail = _tail(st.stdout_tail + "\n" + proc2.stdout)
            st.stderr_tail = _tail(st.stderr_tail + "\n" + proc2.stderr)
            st.eval_ok = (proc2.returncode == 0)
            st.eval_dir = str((batch_dir / "_eval").resolve())
            if not st.eval_ok:
                st.error = (st.error + "\n" if st.error else "") + f"eval-batch failed with returncode={proc2.returncode}"
        except Exception as e:
            st.eval_ok = False
            st.error = (st.error + "\n" if st.error else "") + str(e)
            st.traceback = (st.traceback + "\n" if st.traceback else "") + traceback.format_exc()

        # Collect metrics for report
        try:
            st.metrics = _collect_stage_metrics(batch_dir)
        except Exception:
            st.metrics = {}

        st.finished_at = datetime.datetime.now().isoformat(timespec="seconds")
        dump_json(stages_dir / f"{stage_id}_{name}.json", st.to_dict())
        stage_results.append(st)
        successful_batch_dirs.append(batch_dir)

    # Compare across successful stages
    compare_dir = ensure_dir(suite_root / "compare")
    compare_ok = False
    compare_err = ""
    if len(successful_batch_dirs) >= 2:
        cmd_cmp = [
            sys.executable,
            "-u",
            str(run_py_path),
            "eval-compare",
            "--out-dir", str(compare_dir),
            "--baseline-mode", str(getattr(args, "suite_baseline_mode", "M2") or "M2"),
        ]
        for bd in successful_batch_dirs:
            cmd_cmp += ["--batch-dir", str(bd)]
        try:
            print("\n[SUITE] eval-compare:", " ".join(cmd_cmp))
            proc3 = _run_subprocess(cmd_cmp, cwd=project_root)
            compare_ok = (proc3.returncode == 0)
            if not compare_ok:
                compare_err = _tail(proc3.stderr)
        except Exception as e:
            compare_ok = False
            compare_err = str(e)
    else:
        compare_err = "skip compare: need >=2 successful stages"

    # Manifest
    manifest = {
        "suite_id": suite_id,
        "suite_run_id": suite_run_id,
        "suite_spec": suite_spec,
        "batch_file": str(Path(args.batch_file).resolve()),
        "agent_config": str(Path(args.agent_config).resolve()),
        "outputs_root": str(Path(args.outputs_root).resolve()),
        "library_root": str(Path(args.library_root).resolve()),
        "runs_dir": str(Path(args.runs_dir).resolve()),
        "workers": int(args.workers) if args.workers is not None else None,
        "m8_ks": ks,
        "judge": enable_judge,
        "compare_ok": compare_ok,
        "compare_dir": str(compare_dir.resolve()),
        "compare_error": compare_err,
        "stages": [st.to_dict() for st in stage_results],
    }
    dump_json(suite_root / "suite_manifest.json", manifest)

    # Write report
    _write_suite_report_md(suite_root, suite_id=suite_id, suite_run_id=suite_run_id, stages=stage_results)

    print(f"[SUITE] suite done. suite_root={suite_root.resolve()}")
    print(f"[SUITE] report: {(suite_root / 'suite_report.md').resolve()}")
