from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_provenance(cwd: Path) -> dict[str, Any]:
    return {
        "commit": _run_git(["rev-parse", "HEAD"], cwd),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "status_short": _run_git(["status", "--short"], cwd),
    }


def write_run_metadata(
    run_dir: Path,
    project_root: Path,
    command: list[str],
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": command,
        "config": config,
        "git": git_provenance(project_root),
        "python": sys.version,
        "platform": platform.platform(),
    }
    if extra:
        metadata.update(extra)
    output = run_dir / "run_metadata.json"
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output

