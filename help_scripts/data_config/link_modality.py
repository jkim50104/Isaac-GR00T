#!/usr/bin/env python3
"""
Fix (or create) modality.json symlinks inside local meta directories.

Goal:
- For every <BASE_LOCAL>/<exp>/meta/ directory that exists locally:
  - Ensure modality.json exists and is a symlink to:
      <WORKSPACE_ROOT>/help_scripts/data_config/ai_worker_modality.json
  - If modality.json exists but:
      * is not a symlink, OR
      * points somewhere else, OR
      * is a broken symlink
    => replace it with the correct symlink.
- If modality.json is missing => create correct symlink.

No remote access; local-only.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------- CONFIG YOU MAY EDIT ----------
# Local data root where experiments live (contains exp/meta/)
BASE_LOCAL = Path("data/jkim50104")

# Relative target (from workspace root) to which modality.json must link
TARGET_REL = Path("help_scripts/data_config/ai_worker_modality.json")

# Optional filter for experiments
# None => include all; or e.g. ("ffw_",)
EXPERIMENT_PREFIXES: tuple[str, ...] | None = None

DRY_RUN = False  # True => print actions only, do not modify filesystem
# ---------------------------------------


def _prefix_ok(name: str) -> bool:
    if EXPERIMENT_PREFIXES is None:
        return True
    return any(name.startswith(p) for p in EXPERIMENT_PREFIXES)


def workspace_root() -> Path:
    """
    Assume this script is run somewhere inside the workspace.
    We walk up until we find 'help_scripts' dir.
    If not found, fall back to current working directory.
    """
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / "help_scripts").is_dir():
            return p
    return cur


def is_correct_symlink(link_path: Path, target_abs: Path) -> bool:
    if not link_path.exists() and not link_path.is_symlink():
        return False
    if not link_path.is_symlink():
        return False
    try:
        resolved = link_path.resolve(strict=False)
    except Exception:
        return False
    # Compare fully resolved absolute paths (strict=False allows broken link resolution attempt)
    return resolved == target_abs


def ensure_symlink(link_path: Path, target_abs: Path) -> str:
    """
    Ensure link_path is a symlink pointing to target_abs.
    Returns a status string describing what happened.
    """
    # Ensure parent exists
    if not link_path.parent.is_dir():
        return "SKIP(no_meta_dir)"

    # If already correct, do nothing
    if is_correct_symlink(link_path, target_abs):
        return "OK"

    # Otherwise replace/create
    action_parts = []

    if link_path.exists() or link_path.is_symlink():
        action_parts.append("REPLACE")
        if not DRY_RUN:
            # unlink removes file or symlink; for dirs we'd need rmtree, but modality.json should not be a dir
            link_path.unlink()
    else:
        action_parts.append("CREATE")

    # Create symlink; prefer relative link for portability if possible
    # We attempt to make target relative to link_path.parent
    try:
        rel_target = os.path.relpath(str(target_abs), start=str(link_path.parent.resolve()))
    except Exception:
        rel_target = str(target_abs)

    action_parts.append(f"-> {rel_target}")

    if not DRY_RUN:
        link_path.symlink_to(rel_target)

    return " ".join(action_parts)


def main():
    ws = workspace_root()
    target_abs = (ws / TARGET_REL).resolve()

    print(f"[DEBUG] WORKSPACE_ROOT = {ws}")
    print(f"[DEBUG] BASE_LOCAL      = {BASE_LOCAL.resolve()}")
    print(f"[DEBUG] TARGET         = {target_abs}")
    print(f"[DEBUG] DRY_RUN         = {DRY_RUN}")

    if not target_abs.is_file():
        print("\n[ERROR] Target modality file does not exist:")
        print(f"  {target_abs}")
        print("\nFix: run from the workspace (where help_scripts/ exists) or adjust TARGET_REL.")
        raise SystemExit(1)

    if not BASE_LOCAL.is_dir():
        print("\n[ERROR] BASE_LOCAL does not exist or is not a directory:")
        print(f"  {BASE_LOCAL.resolve()}")
        raise SystemExit(1)

    exps = sorted([p for p in BASE_LOCAL.iterdir() if p.is_dir() and _prefix_ok(p.name)])
    if not exps:
        print("\nNo experiment directories found.")
        return

    total = 0
    changed = 0
    skipped = 0

    for exp_dir in exps:
        meta_dir = exp_dir / "meta"
        if not meta_dir.is_dir():
            skipped += 1
            continue

        link_path = meta_dir / "modality.json"
        total += 1
        status = ensure_symlink(link_path, target_abs)

        exp_name = exp_dir.name
        if status == "OK":
            print(f"[OK]      {exp_name}/meta/modality.json")
        elif status.startswith("SKIP"):
            skipped += 1
            print(f"[SKIP]    {exp_name}  ({status})")
        else:
            changed += 1
            print(f"[FIXED]   {exp_name}/meta/modality.json  {status}")

    print("\n=== SUMMARY ===")
    print(f"Checked meta dirs : {total}")
    print(f"Fixed/created     : {changed}")
    print(f"Skipped           : {skipped}")
    if DRY_RUN:
        print("\n(DRY_RUN enabled: no filesystem changes were made.)")


if __name__ == "__main__":
    main()
