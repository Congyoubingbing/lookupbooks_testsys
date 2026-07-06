from __future__ import annotations
import json
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

INVALID_WIN_CHARS = r'<>:"/\\|?*'

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def dump_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path: Path, default: Any=None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def sanitize_filename(name: str, repl: str="_") -> str:
    out = []
    for ch in name:
        if ch in INVALID_WIN_CHARS or ord(ch) < 32:
            out.append(repl)
        else:
            out.append(ch)
    s = "".join(out).strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # avoid empty
    return s if s else "untitled"


def safe_output_filename(
    name: str,
    *,
    max_len: int = 80,
    add_hash: bool = True,
    sanitize: bool = True,
    repl: str = "_",
) -> str:
    """Generate a Windows-safe, length-bounded filename component.

    This function is designed for chapter titles embedded into output filenames.
    It:
    - optionally sanitizes invalid Windows filename characters
    - strips trailing spaces/dots (invalid on Windows)
    - bounds length; if truncated, can append a short stable hash suffix
    """
    src = "" if name is None else str(name)
    base = sanitize_filename(src, repl=repl) if sanitize else src
    # Windows does not allow trailing space/dot.
    base = base.rstrip(" .")
    base = base if base else "untitled"

    try:
        ml = int(max_len)
    except Exception:
        ml = 80
    if ml <= 0:
        return base
    if len(base) <= ml:
        return base

    if not add_hash:
        return base[:ml].rstrip(" .") or "untitled"

    h = hashlib.sha1(src.encode("utf-8", errors="ignore")).hexdigest()[:8]
    suffix = "__" + h
    keep = ml - len(suffix)
    if keep <= 0:
        return h
    head = base[:keep].rstrip(" .")
    head = head if head else "untitled"
    return head + suffix

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def env_keys(prefix: str, n: int) -> List[str]:
    """Collect API keys from environment.

    Historically we used 1-based indexing (PREFIX1..PREFIXn).
    Some deployments prefer 0-based indexing (PREFIX0..PREFIX(n-1)).

    Rule:
      - If PREFIX0 exists, use 0..n-1
      - else use 1..n
    """
    keys: List[str] = []
    use_zero_based = bool(os.getenv(f"{prefix}0"))
    if use_zero_based:
        idxs = range(0, max(0, int(n)))
    else:
        idxs = range(1, max(0, int(n)) + 1)
    for i in idxs:
        k = os.getenv(f"{prefix}{i}")
        if k:
            keys.append(k.strip())
    return keys

def roman_to_int(s: str) -> Optional[int]:
    # Basic roman numeral converter (i, ii, iv, v, vi, ...)
    if not s:
        return None
    s = s.strip().lower()
    if not re.fullmatch(r"[ivxlcdm]+", s):
        return None
    vals = {'i':1,'v':5,'x':10,'l':50,'c':100,'d':500,'m':1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        v = vals[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total if total > 0 else None

def normalize_title(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[Cc]hapter\s+\d+\s*", "", s)
    s = re.sub(r"^第\s*\d+\s*章\s*", "", s)
    return s.strip().lower()


def compute_code_hash(project_root: Path, *,
    include_exts: Tuple[str, ...] = (".py", ".yaml", ".yml", ".md", ".txt"),
    exclude_dirs: Tuple[str, ...] = ("outputs", "_tmp_pairs", ".git", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"),
) -> str:
    """Compute a stable content hash for the code/config package.

    - Excludes runtime data folders like outputs/ and _tmp_pairs/
    - Includes file relative path + bytes to reduce accidental collisions
    """
    project_root = Path(project_root).resolve()
    files: List[Path] = []
    for p in project_root.rglob("*"):
        try:
            if not p.is_file():
                continue
            if p.suffix.lower() not in include_exts:
                continue
            if any(part in exclude_dirs for part in p.parts):
                continue
            files.append(p)
        except Exception:
            continue

    h = hashlib.sha1()
    for p in sorted(files, key=lambda x: str(x.relative_to(project_root)).lower()):
        rel = str(p.relative_to(project_root)).replace("\\", "/")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        try:
            h.update(p.read_bytes())
        except Exception:
            # fall back to text read
            try:
                h.update(p.read_text(encoding="utf-8", errors="ignore").encode("utf-8"))
            except Exception:
                pass
        h.update(b"\0")
    return h.hexdigest()
