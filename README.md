# lookupbooks_testsys

`lookupbooks_testsys` is a structured textbook-consultation benchmark system for evaluating whether a large language model improves on polymer-theory reasoning after controlled access to a book library. The repository contains the runnable code, core configuration files, prompts, tools, and a Markdown-formatted question set.

## Repository status

This version is prepared for public GitHub upload:

- real API keys and local key files have been removed;
- only the core configuration files are kept under `config/`;
- explanatory Markdown files have been consolidated into this `README.md`;
- runtime outputs, local corpora, generated libraries, caches, logs, and environment files are excluded by `.gitignore`.

## Main workflow

The question-answering workflow follows a Step0--Step6 evidence-access protocol:

1. **Step0**: produce a direct answer (`answer1`).
2. **Step0 judge**: optionally judge whether retrieval is needed.
3. **Step1**: select candidate books from `summarybook.txt`.
4. **Step2**: select one or more books for deeper inspection.
5. **Step3**: select candidate chapters from the selected books.
6. **Step4**: select final chapters for inspection.
7. **Step5**: select final sections; section text is the smallest retrieval unit.
8. **Step6**: answer with retrieved evidence (`answer2`), compare `answer1` and `answer2`, and generate reports.

Step0 and Step6 use a format-validation chain: generation, deterministic validation, minimal repair, optional re-answering, and final normalization.

## Directory layout

```text
config/                  Core YAML configuration files
lookupbooks_sys/          Book-library QA workflow, evaluation, reporting, suite logic
pdf_txt_align/            PDF/TXT-or-TeX alignment and book-structure extraction utilities
prompts/                  Expert and judge prompts required by the system
tests/                    Markdown question set in q0 block format
tools/                    Maintenance, batch-running, repair, and self-check utilities
run.py                    Main command-line entry point
agent_run.py              Backward-compatible wrapper around run.py
requirements.txt          Python dependencies
```

Generated or local-only directories such as `outputs/`, `runs/`, `pairs/`, and generated `library/` contents are intentionally not tracked.

## Environment

Recommended runtime: Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

The default configuration uses an OpenAI-compatible DashScope endpoint and reads API keys from environment variables. Do not place real keys in repository files.

PowerShell example:

```powershell
$env:QWEN_API_KEY_0="replace_with_your_key"
$env:QWEN_API_KEY_1="replace_with_another_key"
```

Bash example:

```bash
export QWEN_API_KEY_0="replace_with_your_key"
export QWEN_API_KEY_1="replace_with_another_key"
```

Key loading rule: if `QWEN_API_KEY_0` exists, the system reads zero-based keys `QWEN_API_KEY_0..QWEN_API_KEY_{N-1}`; otherwise it reads one-based keys `QWEN_API_KEY_1..QWEN_API_KEY_N`. `N` is controlled by `models.key_env_count` in the active YAML config.

For a no-API local smoke test, set:

```bash
export LOOKUPBOOKS_SYS_MOCK_LLM=1
```

## Core configuration files

Only two configuration files are retained:

- `config/default.yaml`: PDF/TXT-or-TeX alignment, TOC parsing, page-label detection, chapter/section boundary extraction, text normalization, and verification settings.
- `config/agent.yaml`: QA, retrieval, model routing, judge, format validation, evidence budget, and evaluation settings.

Important configuration groups:

- `models`: endpoint, model names, timeout, retry policy, environment-key prefix and key count.
- `runtime`: output directory, cache/log directory names, Windows-safe path limits.
- `pdf`, `toc`, `anchors`, `text`, `align`, `verify`: controls for PDF parsing and text alignment.
- `qa`: Step0--Step6 behavior, retrieval limits, M-mode behavior, format validation, evidence budgets, repair policy, report settings.
- `qa_models`: model assignment for direct answer, ranking, final answer, comparison, formatting, and self-assessment.
- `eval`: judge model, judge workers, rubric path, and maximum text lengths for evaluation inputs.

## Input data conventions

### PDF/TXT-or-TeX pairs

For book extraction, place matched source files under a local `pairs/` directory:

```text
pairs/
  BookA.pdf
  BookA.tex   # or BookA.txt
  BookB.pdf
  BookB.txt
```

`pairs/` is ignored by Git because it may contain copyrighted or private source material.

### Batch questions

Batch questions are Markdown files containing one q0 code block per question:

````markdown
## Q001｜Question title

```q0
Question text here.
```
````

The included batch file is `tests/批量测试_最终版v3_q0格式.md`.

## Main commands

Use `run.py` as the primary entry point.

List matched PDF/TXT-or-TeX pairs:

```bash
python run.py list --pairs-dir ./pairs
```

Run PDF/text alignment and chapter/section extraction:

```bash
python run.py align --pairs-dir ./pairs --config ./config/default.yaml --workers 5
```

Build a library from extracted outputs:

```bash
python run.py build-library \
  --outputs-root ./outputs \
  --library-root ./library \
  --agent-config ./config/agent.yaml
```

Build the library without generating summaries:

```bash
python run.py build-library \
  --outputs-root ./outputs \
  --library-root ./library \
  --agent-config ./config/agent.yaml \
  --skip-summarize
```

Generate or refresh book/chapter summaries:

```bash
python run.py summarize \
  --agent-config ./config/agent.yaml \
  --outputs-root ./outputs \
  --library-root ./library
```

List library entries:

```bash
python run.py library-list --library-root ./library
```

Ask a single question:

```bash
python run.py ask \
  --agent-config ./config/agent.yaml \
  --outputs-root ./outputs \
  --library-root ./library \
  --question-file ./tests/q0.txt \
  --test-mode M3
```

Run a batch:

```bash
python run.py batch-ask \
  --batch-file ./tests/批量测试_最终版v3_q0格式.md \
  --agent-config ./config/agent.yaml \
  --outputs-root ./outputs \
  --library-root ./library \
  --runs-dir ./runs/batch \
  --workers 5 \
  --test-mode M3
```

Evaluate a batch with programmatic metrics only:

```bash
python run.py eval-batch \
  --batch-dir ./runs/batch/<batch_id> \
  --agent-config ./config/agent.yaml
```

Evaluate a batch with LLM-as-judge scoring:

```bash
python run.py eval-batch \
  --batch-dir ./runs/batch/<batch_id> \
  --agent-config ./config/agent.yaml \
  --judge \
  --skip-existing
```

Compare multiple batch runs:

```bash
python run.py eval-compare \
  --out-dir ./runs/_compare/compare_<timestamp> \
  --baseline-mode M2 \
  --batch-dir ./runs/batch/<id_M1> \
  --batch-dir ./runs/batch/<id_M2> \
  --batch-dir ./runs/batch/<id_M3>
```

Dump the loaded configuration:

```bash
python run.py config-dump --config ./config/agent.yaml
```

Backward-compatible wrappers are retained but not preferred:

```bash
python agent_run.py <subcommand> ...
python tools/run_batch_q0.py ...
python tools/score_batch_results.py ...
```

## Test modes

`ask` and `batch-ask` accept `--test-mode` to override `qa.test_mode` in `config/agent.yaml`.

| Mode | Purpose |
|---|---|
| M1 | Forced structured textbook retrieval. |
| M2 | No retrieval baseline; direct answer only. |
| M3 | Adaptive retrieval with Step0 judge. |
| M4 | No retrieval plus expert prompt. |
| M5 | Confidence-gated retrieval. |
| M6 | Chapter-summary-only ablation. |
| M7 | Skip section-level LLM ranking and use rule-based section filling. |
| M8 | Section-budget sweep. |
| M9 | Section-level retrieval with expert prompt. |

For M8, use `--topk-sections` to control section budget:

```bash
python run.py batch-ask \
  --batch-file ./tests/批量测试_最终版v3_q0格式.md \
  --agent-config ./config/agent.yaml \
  --outputs-root ./outputs \
  --library-root ./library \
  --workers 5 \
  --test-mode M8 \
  --topk-sections 3
```

A suite run can execute multiple modes and M8 budgets:

```bash
python run.py batch-ask \
  --batch-file ./tests/批量测试_最终版v3_q0格式.md \
  --agent-config ./config/agent.yaml \
  --outputs-root ./outputs \
  --library-root ./library \
  --workers 5 \
  --suite M1-M8 \
  --suite-id smoke_testsys_small \
  --suite-m8-ks "1,3,5" \
  --suite-judge
```

## Evaluation outputs

`eval-batch` writes results under `runs/batch/<batch_id>/_eval/`:

- `auto_metrics.jsonl` and `auto_metrics.csv`: programmatic metrics;
- `judge_outputs/<Qxxx>.json`: raw and parsed judge output, when `--judge` is used;
- `judge_scores.jsonl` and `judge_scores.csv`: structured judge scores, when `--judge` is used;
- `combined.csv`: merged per-question evaluation table;
- `summary.json`: aggregate summary;
- `eval_report.md`: Markdown evaluation report.

Programmatic metrics cover identity, mode, execution status, retrieval behavior, format compliance, evidence budget, selected-integrity issues, and step durations. Judge output follows the `judge_v1` JSON schema with integer 0--10 scores for `overall`, `correctness`, `completeness`, `derivation`, `clarity`, `grounding`, and `hallucination_resistance`.

## Self-check

Run the local smoke test before committing changes:

```bash
python tools/self_check.py
```

The self-check uses `LOOKUPBOOKS_SYS_MOCK_LLM=1`, compiles/imports modules, builds a synthetic library, runs M1--M9 batch QA, evaluates results, refreshes reports, and aggregates repeats without real API calls.

## Security and GitHub upload checklist

Before publishing:

1. Run a secret scan, especially for `sk-`, `api_key`, `secret`, `token`, and local credential files.
2. Confirm `config/` contains only `agent.yaml` and `default.yaml`.
3. Do not commit `outputs/`, `runs/`, `pairs/`, generated `library/` data, `_cache/`, `_logs/`, or real `.env` files.
4. Keep real API keys only in environment variables or an untracked secret manager.
5. Run `python tools/self_check.py` after code edits.
