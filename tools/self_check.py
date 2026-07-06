import os, sys, pkgutil, importlib, inspect, compileall, json, shutil, subprocess, tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def fail(msg: str) -> None:
    print("FAIL:", msg)
    raise SystemExit(1)


def run_cmd(cmd, *, env=None):
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        fail(
            "Command failed: "
            + " ".join(cmd)
            + "\n--- stdout ---\n"
            + (r.stdout or "")
            + "\n--- stderr ---\n"
            + (r.stderr or "")
        )
    return r


def mock_e2e_cli_test() -> None:
    env = os.environ.copy()
    env["LOOKUPBOOKS_SYS_MOCK_LLM"] = "1"
    tmp_root = tempfile.mkdtemp(prefix="lookupbooks_selfcheck_")
    shutil.rmtree(os.path.join(PROJECT_ROOT, "runs", "batch"), ignore_errors=True)
    shutil.rmtree(os.path.join(PROJECT_ROOT, "runs", "ask"), ignore_errors=True)
    shutil.rmtree(os.path.join(PROJECT_ROOT, "runs", "_repeat_agg"), ignore_errors=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "runs", "batch"), exist_ok=True)
    outputs_root = os.path.join(tmp_root, "outputs")
    library_root = os.path.join(tmp_root, "library")
    tests_root = os.path.join(tmp_root, "tests")
    os.makedirs(outputs_root, exist_ok=True)
    os.makedirs(tests_root, exist_ok=True)

    def write_book(book_dir_name: str, title: str, chapter_title: str, section_title: str):
        root = os.path.join(outputs_root, book_dir_name)
        ch_dir = os.path.join(root, "chapters")
        sec_dir = os.path.join(root, "sections", "ch01")
        os.makedirs(ch_dir, exist_ok=True)
        os.makedirs(sec_dir, exist_ok=True)
        with open(os.path.join(ch_dir, "ch01_01.txt"), "w", encoding="utf-8") as f:
            f.write("\\chapter{" + chapter_title + "}\n" + ((section_title + " related content. ") * 30) + "\n")
        with open(os.path.join(sec_dir, "s1.txt"), "w", encoding="utf-8") as f:
            f.write(((section_title + " detailed section content. ") * 40) + "\n")
        with open(os.path.join(root, "book_overview.json"), "w", encoding="utf-8") as f:
            json.dump({"book_title": title, "book_summary": f"{title} 概述。"}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(root, "chapter_overview.json"), "w", encoding="utf-8") as f:
            json.dump({
                "chapters": [{
                    "chapter_id": "c1",
                    "chapter_no": 1,
                    "chapter_title": chapter_title,
                    "chapter_summary": f"{chapter_title} 概述。",
                    "keywords": ["blob"],
                    "section_index_file": "sections/ch01/section_index.json",
                    "chapter_file": "chapters/ch01_01.txt",
                }]}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(sec_dir, "section_index.json"), "w", encoding="utf-8") as f:
            json.dump({"items": [{
                "section_id": "s1",
                "title": section_title,
                "summary": f"{section_title} 摘要。",
                "keywords": ["blob"],
                "file": "s1.txt",
                "exposure_decision": "expose",
                "quality_label": "good",
                "quality_score": 1.0,
            }]}, f, ensure_ascii=False, indent=2)

    write_book("BookA", "Book A", "Chapter A", "Blob basics")
    write_book("BookB", "Book B", "Chapter B", "Polymer notes")

    run_py = os.path.join(PROJECT_ROOT, "run.py")
    agent_cfg = os.path.join(PROJECT_ROOT, "config", "agent.yaml")

    run_cmd([sys.executable, run_py, "build-library", "--outputs-root", outputs_root, "--library-root", library_root, "--agent-config", agent_cfg], env=env)

    batch_file = os.path.join(tests_root, "batch.md")
    with open(batch_file, "w", encoding="utf-8") as f:
        f.write("# batch\n\n## Q01｜Test\n\n```q0\nTRUNCATION_TEST: What is blob?\n```\n")

    modes = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
    batch_dirs = []
    for mode in modes:
        batch_id = f"selfcheck_{mode.lower()}"
        cmd = [
            sys.executable, run_py, "batch-ask",
            "--batch-file", batch_file,
            "--agent-config", agent_cfg,
            "--outputs-root", outputs_root,
            "--library-root", library_root,
            "--workers", "1",
            "--test-mode", mode,
            "--batch-id", batch_id,
        ]
        if mode == "M8":
            cmd += ["--topk-sections", "3"]
        run_cmd(cmd, env=env)
        batch_dir = os.path.join(PROJECT_ROOT, "runs", "batch", batch_id)
        batch_dirs.append(batch_dir)
        qa_file = os.path.join(batch_dir, "Q01", "qa_result.json")
        if not os.path.exists(qa_file):
            fail(f"{mode}: qa_result.json missing")
        qa = json.load(open(qa_file, "r", encoding="utf-8"))
        if mode == "M6" and not qa.get("chapter_only_mode"):
            fail("M6: chapter_only_mode should be true")
        if mode == "M2" and qa.get("retrieval_used"):
            fail("M2: retrieval_used should be false")
        rep = open(os.path.join(batch_dir, "Q01", "report.md"), "r", encoding="utf-8").read()
        for heading in ["### 题目", "### 检索内容", "### 关键推导步骤", "### 实际的具体解题过程", "### 最终答案", "### 评分"]:
            if heading not in rep:
                fail(f"{mode}: missing report heading {heading}")

        run_cmd([sys.executable, run_py, "eval-batch", "--batch-dir", batch_dir, "--agent-config", agent_cfg, "--judge", "--judge-workers", "1"], env=env)
        rep2 = open(os.path.join(batch_dir, "Q01", "report.md"), "r", encoding="utf-8").read()
        if "- overall:" not in rep2:
            fail(f"{mode}: score section was not refreshed after eval-batch --judge")
        import csv
        with open(os.path.join(batch_dir, '_eval', 'combined.csv'), 'r', encoding='utf-8') as f:
            row = next(csv.DictReader(f))
        if row.get('judge_score_overall', '') == '':
            fail(f"{mode}: combined.csv missing judge_score_overall after eval aggregation")

    run_cmd([sys.executable, run_py, "refresh-reports", "--batch-dir", batch_dirs[0]], env=env)
    out_dir = os.path.join(PROJECT_ROOT, "runs", "_repeat_agg", "selfcheck")
    cmd = [sys.executable, run_py, "aggregate-repeats", "--out-dir", out_dir]
    for bd in batch_dirs:
        cmd += ["--batch-dir", bd]
    run_cmd(cmd, env=env)
    if not os.path.exists(os.path.join(out_dir, "repeat_agg.csv")):
        fail("aggregate-repeats: repeat_agg.csv missing")
    shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> None:
    if not compileall.compile_dir(PROJECT_ROOT, quiet=1):
        fail("compileall failed")
    import pdf_txt_align
    pkg_dir = os.path.dirname(pdf_txt_align.__file__)
    print("pdf_txt_align loaded from:", pdf_txt_align.__file__)
    for m in pkgutil.iter_modules([pkg_dir]):
        importlib.import_module("pdf_txt_align." + m.name)
    from pdf_txt_align import llm_calls, verify
    sig = inspect.signature(llm_calls.vl_read_page_label)
    if "crop_policy" not in sig.parameters:
        fail("llm_calls.vl_read_page_label missing crop_policy")
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        fail("llm_calls.vl_read_page_label missing **kwargs")
    vsig = inspect.signature(verify.verify_chapter_segments)
    if "match_text" not in vsig.parameters:
        fail("verify.verify_chapter_segments missing match_text")
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in vsig.parameters.values()):
        fail("verify.verify_chapter_segments missing **kwargs")
    if not isinstance(verify._norm("Hello  World"), str):
        fail("verify._norm should return str")
    import lookupbooks_sys as lbs
    if not getattr(lbs, "__version__", ""):
        fail("lookupbooks_sys.__version__ missing")
    mock_e2e_cli_test()
    print("OK: compileall + imports + signature checks + mock E2E CLI (M1-M9 + judge + refresh + aggregate-repeats)")


if __name__ == "__main__":
    main()
