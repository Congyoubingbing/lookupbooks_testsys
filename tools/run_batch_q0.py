# -*- coding: utf-8 -*-
"""tools/run_batch_q0.py (deprecated wrapper)

历史版本中，这个脚本会：
- 解析 Markdown 中的 ```q0 ...``` 块
- 逐题调用 `python run.py ask --question-file ...`（用于绕过 Windows 命令行长度限制）

自 v0.2.0 起，`run.py batch-ask` 已在进程内直接读取题干并写入每题目录的 q0.txt，
不会受到命令行长度限制，因此原实现逻辑属于冗余。

本脚本保留为“兼容入口”，仅把参数透传给：
  python run.py batch-ask ...

注意：
- 旧参数 --out-dir 若提供，则会被拆分为：--runs-dir=<parent> + --batch-id=<name>
- 其余参数尽量保持兼容；更完整功能请直接使用 run.py。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="(deprecated) wrapper for `python run.py batch-ask`.")
    ap.add_argument("--batch-file", required=True, help="Markdown file containing ```q0 blocks")
    ap.add_argument("--tag", default="q0", help="Code block tag name (default: q0)")
    ap.add_argument("--outputs-root", required=True)
    ap.add_argument("--library-root", required=True)
    ap.add_argument("--agent-config", required=True)

    # compatibility options
    ap.add_argument("--out-dir", default=None, help="(compat) full output dir, e.g. runs/batch/<batch_id>")
    ap.add_argument("--runs-dir", default=None, help="(compat) runs dir (parent of batch_id)")
    ap.add_argument("--batch-id", default=None, help="(compat) fixed batch_id")

    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--auto-summarize", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--test-mode", default=None)
    ap.add_argument("--topk-books", type=int, default=None)
    ap.add_argument("--topk-pairs", type=int, default=None)

    args = ap.parse_args()

    # Resolve out-dir -> runs-dir + batch-id
    runs_dir = args.runs_dir
    batch_id = args.batch_id
    if args.out_dir:
        od = Path(args.out_dir)
        runs_dir = str(od.parent)
        batch_id = str(od.name)

    cmd = [
        sys.executable,
        "run.py",
        "batch-ask",
        "--batch-file",
        str(args.batch_file),
        "--tag",
        str(args.tag),
        "--outputs-root",
        str(args.outputs_root),
        "--library-root",
        str(args.library_root),
        "--agent-config",
        str(args.agent_config),
        "--start",
        str(int(args.start)),
        "--end",
        str(int(args.end)),
    ]

    if runs_dir:
        cmd += ["--runs-dir", str(runs_dir)]
    if batch_id:
        cmd += ["--batch-id", str(batch_id)]
    if args.workers is not None:
        cmd += ["--workers", str(int(args.workers))]
    if args.stop_on_error:
        cmd += ["--stop-on-error"]
    if args.auto_summarize:
        cmd += ["--auto-summarize"]
    if args.verbose:
        cmd += ["--verbose"]
    if args.test_mode:
        cmd += ["--test-mode", str(args.test_mode)]
    if args.topk_books is not None:
        cmd += ["--topk-books", str(int(args.topk_books))]
    if args.topk_pairs is not None:
        cmd += ["--topk-pairs", str(int(args.topk_pairs))]

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
