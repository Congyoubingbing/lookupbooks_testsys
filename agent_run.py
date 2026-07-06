#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""agent_run.py (compat wrapper)

v0.2.0 起，推荐只使用统一入口 run.py：
  python run.py align|build-library|summarize|ask

此脚本仅作为兼容入口，等价于执行 run.py。
"""

from run import main

if __name__ == "__main__":
    main()
