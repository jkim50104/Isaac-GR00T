#!/usr/bin/env python3
"""Pull all experiment datasets from lunar server to local (missing files only)."""
import subprocess
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
REMOTE = "lunar"
BASE_REMOTE = "/home/robi/projects/Isaac-GR00T/data/jkim50104"
BASE_LOCAL = Path("data/jkim50104")

REQUIRED_SUBDIRS = ("data", "meta", "videos")
META_EXCLUDES = ("relative_stats.json", "stats.json", "modality.json")

DRY_RUN = False  # if True, rsync uses -n only


# ---------------------------
# Helpers
# ---------------------------
def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def ssh_lines(cmd: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ["ssh", REMOTE, cmd],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        print(f"[SSH ERROR] cmd={cmd}\n{e.output}")
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def list_remote_dirs(path: str) -> list[str]:
    ok = ssh_lines(f'cd "{path}" 2>/dev/null && echo OK || echo NO')
    if not ok or ok[-1] != "OK":
        return []
    cmd = (
        f'cd "{path}" 2>/dev/null || exit 0; '
        f'for x in *; do [ -d "$x" ] && echo "$x"; done'
    )
    dirs = ssh_lines(cmd)
    dirs.sort()
    return dirs


def _rsync_exclude_args(subdir: str) -> list[str]:
    if subdir != "meta":
        return []
    args: list[str] = []
    for f in META_EXCLUDES:
        args += ["--exclude", f]
    return args


def rsync_pull(exp: str, sd: str) -> None:
    src = f"{REMOTE}:{BASE_REMOTE}/{exp}/{sd}/"
    dst_path = (BASE_LOCAL / exp / sd).resolve()
    dst_path.mkdir(parents=True, exist_ok=True)

    cmd = ["rsync", "-a", "--ignore-existing", "--info=progress2", "--partial"]
    cmd += _rsync_exclude_args(sd)
    if DRY_RUN:
        cmd.append("-n")
    cmd += [src, str(dst_path) + "/"]
    run(cmd)


# ---------------------------
# Main
# ---------------------------
def main():
    print(f"Pull from: {REMOTE}:{BASE_REMOTE}")
    print(f"Local:     {BASE_LOCAL.resolve()}")
    print()

    remote_exps = sorted(list_remote_dirs(BASE_REMOTE))
    if not remote_exps:
        print("No experiments found on remote.")
        return

    local_exps = set()
    if BASE_LOCAL.is_dir():
        local_exps = {p.name for p in BASE_LOCAL.iterdir() if p.is_dir()}

    skipped = [e for e in remote_exps if e in local_exps]
    new_exps = [e for e in remote_exps if e not in local_exps]

    if skipped:
        print(f"Skipping {len(skipped)} existing:")
        for exp in skipped:
            print(f"  [skip] {exp}")
        print()

    if not new_exps:
        print("Nothing new to pull.")
        return

    print(f"Will pull {len(new_exps)} new experiment(s):")
    for exp in new_exps:
        print(f"  [new]  {exp}")

    ans = input("\nPull these? [y/N] ").strip().lower()
    if ans not in {"y", "yes"}:
        print("Aborted.")
        return

    print()
    for exp in new_exps:
        print(f"--- {exp} ---")
        for sd in REQUIRED_SUBDIRS:
            rsync_pull(exp, sd)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
