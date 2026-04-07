#!/usr/bin/env python3
"""
Pull lerobot datasets from lunar into local data directory.

Source: lunar:/home/robi/projects/physical_ai_tools/docker/huggingface/lerobot/jkim50104/
Dest:   data/jkim50104/

Each experiment folder (e.g. ffw_sg2_rev1_clear_item) is expected to contain
the standard lerobot subdirs: data/, meta/, videos/.
"""
import subprocess
from pathlib import Path
import sys

# ---------------------------
# Config
# ---------------------------
REMOTE = "lunar"
REMOTE_BASE = "/home/robi/projects/physical_ai_tools/docker/huggingface/lerobot/jkim50104"
LOCAL_BASE = Path("data/jkim50104")

REQUIRED_SUBDIRS = ("data", "meta", "videos")

# Exclude these files from meta/ sync (they get regenerated locally)
META_EXCLUDES = ("relative_stats.json", "stats.json", "modality.json")

DRY_RUN = False
ENABLE_COPY = True


# ---------------------------
# tqdm fallback
# ---------------------------
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it


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


def remote_subdirs_present(exp: str) -> set[str]:
    present: set[str] = set()
    for sd in REQUIRED_SUBDIRS:
        out = ssh_lines(f'[ -d "{REMOTE_BASE}/{exp}/{sd}" ] && echo OK || echo NO')
        if out and out[-1] == "OK":
            present.add(sd)
    return present


def local_subdirs_present(exp: str) -> set[str]:
    base = LOCAL_BASE / exp
    return {sd for sd in REQUIRED_SUBDIRS if (base / sd).is_dir()}


def _rsync_exclude_args(subdir: str) -> list[str]:
    if subdir != "meta":
        return []
    args: list[str] = []
    for f in META_EXCLUDES:
        args += ["--exclude", f]
    return args


def _rsync_missing_count(src: str, dst: str, subdir: str) -> int:
    cmd = ["rsync", "-a", "--ignore-existing", "-n", "--out-format=%n"]
    cmd += _rsync_exclude_args(subdir)
    cmd += [src, dst]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"[RSYNC DRYRUN ERROR]\ncmd={' '.join(cmd)}\n{e.output}")
        return 0
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    files = [ln for ln in lines if not ln.endswith("/")]
    return len(files)


def rsync_pull(exp: str, sd: str) -> None:
    src = f"{REMOTE}:{REMOTE_BASE}/{exp}/{sd}/"
    dst_path = (LOCAL_BASE / exp / sd).resolve()
    dst_path.mkdir(parents=True, exist_ok=True)

    cmd = ["rsync", "-a", "--ignore-existing", "--info=progress2", "--partial"]
    cmd += _rsync_exclude_args(sd)
    if DRY_RUN:
        cmd.append("-n")
    cmd += [src, str(dst_path) + "/"]
    run(cmd)


# ---------------------------
# UI helpers
# ---------------------------
def _parse_index_set(s: str, n: int) -> set[int]:
    s = s.strip().lower()
    if s in {"a", "all"}:
        return set(range(n))
    if not s:
        return set()
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p]
    out: set[int] = set()
    for p in parts:
        if "-" in p:
            lo, hi = p.split("-", 1)
            try:
                lo_i, hi_i = int(lo), int(hi)
            except ValueError:
                continue
            lo0, hi0 = lo_i - 1, hi_i - 1
            if lo0 > hi0:
                lo0, hi0 = hi0, lo0
            for i in range(lo0, hi0 + 1):
                if 0 <= i < n:
                    out.add(i)
        else:
            try:
                i = int(p) - 1
            except ValueError:
                continue
            if 0 <= i < n:
                out.add(i)
    return out


def confirm(prompt: str) -> bool:
    print(f"\n{prompt}")
    print("Type 'yes' to proceed, anything else to abort:")
    return input("> ").strip().lower() == "yes"


# ---------------------------
# Main
# ---------------------------
def main():
    print(f"[INFO] Source:  {REMOTE}:{REMOTE_BASE}")
    print(f"[INFO] Dest:    {LOCAL_BASE.resolve()}")

    # List what's on lunar
    remote_exps = list_remote_dirs(REMOTE_BASE)
    if not remote_exps:
        print("\nNo experiment folders found on remote.")
        return

    # Build preview: for each remote experiment, show what would be pulled
    preview_lines: list[str] = []
    exp_list: list[str] = []

    for exp in tqdm(remote_exps, desc="Scanning remote experiments", unit="exp"):
        r_subs = remote_subdirs_present(exp)
        l_subs = local_subdirs_present(exp)

        local_status = "EXISTS" if (LOCAL_BASE / exp).is_dir() else "NEW"

        # Count files to pull per subdir
        total_pull = 0
        for sd in REQUIRED_SUBDIRS:
            if sd not in r_subs:
                continue
            src = f"{REMOTE}:{REMOTE_BASE}/{exp}/{sd}/"
            dst_path = LOCAL_BASE / exp / sd
            dst_path.mkdir(parents=True, exist_ok=True)
            total_pull += _rsync_missing_count(src, str(dst_path.resolve()) + "/", sd)

        r_dirs = ",".join(sorted(r_subs)) if r_subs else "NONE"
        tag = "UP_TO_DATE" if total_pull == 0 else f"PULL:{total_pull} files"
        line = f"{exp:<45}  LOCAL:{local_status:<7}  REMOTE_DIRS:{r_dirs:<15}  {tag}"
        preview_lines.append(line)
        exp_list.append(exp)

    # Show menu
    print(f"\n=== LEROBOT DATASETS ON {REMOTE}:{REMOTE_BASE} ===")
    for i, line in enumerate(preview_lines, 1):
        print(f"  [{i:>2}] {line}")

    print("\nSelect by index (e.g. '1 3 5' or '1-3,7') or 'all'.")
    print("Type empty / anything invalid to abort.")
    ans = input("> ").strip()
    idxs = _parse_index_set(ans, len(preview_lines))
    if not idxs:
        print("\nAborted.")
        sys.exit(0)

    selected_exps = [exp_list[i] for i in sorted(idxs)]

    # Build pull plan
    plan: list[tuple[str, str]] = []
    for exp in selected_exps:
        r_subs = remote_subdirs_present(exp)
        for sd in REQUIRED_SUBDIRS:
            if sd in r_subs:
                plan.append((exp, sd))

    if not plan:
        print("\nNothing to pull.")
        return

    # Final preview
    print("\n=== PULL PLAN ===")
    for exp, sd in plan:
        src = f"{REMOTE}:{REMOTE_BASE}/{exp}/{sd}/"
        dst_path = LOCAL_BASE / exp / sd
        dst_path.mkdir(parents=True, exist_ok=True)
        n = _rsync_missing_count(src, str(dst_path.resolve()) + "/", sd)
        print(f"  {exp}/{sd:<6}  -> {n} new files")

    if not ENABLE_COPY:
        print("\nENABLE_COPY is False; exiting.")
        return

    if not confirm("Proceed with pulling?"):
        print("\nAborted.")
        sys.exit(0)

    print("\nStarting pull...\n")
    for exp, sd in tqdm(plan, desc="Pulling", unit="dir"):
        rsync_pull(exp, sd)

    print("\nDone.")


if __name__ == "__main__":
    main()
