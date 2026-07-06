# -*- coding: utf-8 -*-
"""tools/score_batch_results.py (deprecated wrapper)

历史版本中，该脚本对 tools/run_batch_q0.py 产生的 batch_summary.json 做轻量统计。
自 v0.2.0 起，推荐使用统一评测入口：

  python run.py eval-batch --batch-dir runs/batch/<batch_id> --agent-config config/agent.yaml

它会生成：
- _eval/auto_metrics.(jsonl|csv)
- _eval/combined.csv
- _eval/summary.json
- _eval/eval_report.md

本脚本仅作为兼容入口：接收 --batch-summary，并自动推断 batch_dir 后调用 run.py eval-batch。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="(deprecated) wrapper for `python run.py eval-batch`.")
    ap.add_argument("--batch-summary", required=True, help="Path to batch_summary.json")
    ap.add_argument("--agent-config", default="./config/agent.yaml", help="Path to agent config")
    ap.add_argument("--judge", action="store_true", help="Enable LLM-as-a-judge scoring")
    ap.add_argument("--rubric", default="", help="Override rubric markdown path")
    ap.add_argument("--judge-workers", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    bs = Path(args.batch_summary)
    if not bs.exists():
        raise SystemExit(f"not found: {bs}")
    batch_dir = bs.parent

    cmd = [
        sys.executable,
        "run.py",
        "eval-batch",
        "--batch-dir",
        str(batch_dir),
        "--agent-config",
        str(args.agent_config),
    ]
    if args.judge:
        cmd += ["--judge"]
    if args.rubric:
        cmd += ["--rubric", str(args.rubric)]
    if args.judge_workers:
        cmd += ["--judge-workers", str(int(args.judge_workers))]
    if args.skip_existing:
        cmd += ["--skip-existing"]

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
