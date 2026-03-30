#!/usr/bin/env python3
"""
Ensure modality.json in every <BASE_LOCAL>/<exp>/meta/ is a symlink to
help_scripts/data_config/ai_worker_modality.json.
Creates or replaces as needed.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_LOCAL = Path("data/jkim50104")
TARGET_REL = Path("help_scripts/data_config/ai_worker_modality.json")
EXPERIMENT_PREFIXES: tuple[str, ...] | None = None
DRY_RUN = False


def workspace_root() -> Path:
    for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (p / "help_scripts").is_dir():
            return p
    return Path.cwd().resolve()


def ensure_symlink(link_path: Path, target_abs: Path) -> str:
    if not link_path.parent.is_dir():
        return "SKIP"

    # Already correct
    if link_path.is_symlink() and link_path.resolve(strict=False) == target_abs:
        return "OK"

    rel_target = os.path.relpath(str(target_abs), start=str(link_path.parent.resolve()))
    action = "REPLACE" if (link_path.exists() or link_path.is_symlink()) else "CREATE"

    if not DRY_RUN:
        if action == "REPLACE":
            link_path.unlink()
        link_path.symlink_to(rel_target)

    return action


def main():
    ws = workspace_root()
    target_abs = (ws / TARGET_REL).resolve()

    if not target_abs.is_file():
        raise SystemExit(f"[ERROR] Target not found: {target_abs}")
    if not BASE_LOCAL.is_dir():
        raise SystemExit(f"[ERROR] BASE_LOCAL not found: {BASE_LOCAL.resolve()}")

    exps = sorted(
        p for p in BASE_LOCAL.iterdir()
        if p.is_dir() and (EXPERIMENT_PREFIXES is None or any(p.name.startswith(x) for x in EXPERIMENT_PREFIXES))
    )

    ok = changed = skipped = 0
    for exp_dir in exps:
        status = ensure_symlink(exp_dir / "meta" / "modality.json", target_abs)
        label = {"OK": "ok", "SKIP": "skip", "CREATE": "create", "REPLACE": "replace"}[status]
        print(f"  [{label:7}] {exp_dir.name}/meta/modality.json")
        if status == "OK":
            ok += 1
        elif status == "SKIP":
            skipped += 1
        else:
            changed += 1

    print(f"\nok={ok}  fixed={changed}  skipped={skipped}", end="")
    print("  (dry-run)" if DRY_RUN else "")


if __name__ == "__main__":
    main()
