#!/usr/bin/env python3
"""Pull experiment datasets from a remote server to local (missing files only).

Supported remotes: lunar, turing, pearl, rosen, hinton, ai_worker.
ai_worker is a robot server — this script must be run from lunar to reach it.

All datasets in DATASETS are synced (jkim50104, ACS_ROBI).

Examples:
  # Pull all datasets from turing
  python pull_lerobot_data.py turing

  # Pull all datasets from lunar
  python pull_lerobot_data.py lunar

  # Pull all datasets from ai_worker (must be run from lunar)
  python pull_lerobot_data.py ai_worker

  # Dry-run: show what would be pulled without syncing
  python pull_lerobot_data.py pearl -n
"""
import argparse
import socket
import subprocess
import sys
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
SERVERS = ["lunar", "turing", "pearl", "rosen", "hinton", "ai_worker"]
DATASETS = ["jkim50104", "ACS_ROBI"]

# ai_worker can only be reached from this machine.
AI_WORKER_GATEWAY = "lunar"

SERVER_PATHS = {
    "ai_worker": "projects/physical_ai_tools/docker/huggingface/lerobot",
    "hinton": "/data1/jokim/datasets/lerobot",
}
DEFAULT_SERVER_PATH = "projects/Isaac-GR00T/data"

LOCAL_BASE = Path("data")  # relative to work dir

REQUIRED_SUBDIRS = ("data", "meta", "videos")
META_EXCLUDES = ("relative_stats.json", "stats.json", "modality.json")

DRY_RUN = False


# ---------------------------
# Helpers
# ---------------------------
def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def ssh_lines(remote: str, cmd: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ["ssh", remote, cmd],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        print(f"[SSH ERROR] {remote}: {cmd}\n{e.output}")
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def list_remote_dirs(remote: str, path: str) -> list[str]:
    ok = ssh_lines(remote, f'cd "{path}" 2>/dev/null && echo OK || echo NO')
    if not ok or ok[-1] != "OK":
        return []
    cmd = (
        f'cd "{path}" 2>/dev/null || exit 0; '
        f'for x in *; do [ -d "$x" ] && echo "$x"; done'
    )
    dirs = ssh_lines(remote, cmd)
    dirs.sort()
    return dirs


def _rsync_exclude_args(subdir: str) -> list[str]:
    if subdir != "meta":
        return []
    args: list[str] = []
    for f in META_EXCLUDES:
        args += ["--exclude", f]
    return args


def rsync_pull(remote: str, remote_base: str, local_base: Path, exp: str, sd: str) -> None:
    src = f"{remote}:{remote_base}/{exp}/{sd}/"
    dst_path = (local_base / exp / sd).resolve()
    dst_path.mkdir(parents=True, exist_ok=True)

    cmd = ["rsync", "-a", "--ignore-existing", "--info=progress2", "--partial"]
    cmd += _rsync_exclude_args(sd)
    if DRY_RUN:
        cmd.append("-n")
    cmd += [src, str(dst_path) + "/"]
    run(cmd)


def sync_dataset(remote: str, source_base: str, dataset: str) -> None:
    remote_base = f"{source_base}/{dataset}"
    local_base = LOCAL_BASE / dataset

    print(f"=== Dataset: {dataset} ===")
    print(f"Pull from: {remote}:{remote_base}")
    print(f"Local:     {local_base.resolve()}")
    print()

    remote_exps = sorted(list_remote_dirs(remote, remote_base))
    if not remote_exps:
        print("No experiments found on remote.")
        print()
        return

    local_exps = set()
    if local_base.is_dir():
        local_exps = {p.name for p in local_base.iterdir() if p.is_dir()}

    skipped = [e for e in remote_exps if e in local_exps]
    new_exps = [e for e in remote_exps if e not in local_exps]

    if skipped:
        print(f"Skipping {len(skipped)} existing:")
        for exp in skipped:
            print(f"  [skip] {exp}")
        print()

    if not new_exps:
        print("Nothing new to pull.")
        print()
        return

    print(f"Will pull {len(new_exps)} new experiment(s):")
    for exp in new_exps:
        print(f"  [new]  {exp}")
    print()

    for exp in new_exps:
        print(f"--- {exp} ---")
        for sd in REQUIRED_SUBDIRS:
            rsync_pull(remote, remote_base, local_base, exp, sd)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "remote",
        nargs="?",
        choices=SERVERS,
        help=f"Remote server to pull from. Options: {', '.join(SERVERS)}",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Show what would be synced without actually syncing",
    )
    return parser.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()

    if args.remote is None:
        print("Select a server:")
        for i, s in enumerate(SERVERS):
            note = f"  (must be run from {AI_WORKER_GATEWAY})" if s == "ai_worker" else ""
            print(f"  [{i + 1}] {s}{note}")
        choice = input("\nEnter number or name: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(SERVERS):
            args.remote = SERVERS[int(choice) - 1]
        elif choice in SERVERS:
            args.remote = choice
        else:
            print("Invalid selection.")
            sys.exit(1)

    if args.remote == "ai_worker":
        hostname = socket.gethostname()
        if AI_WORKER_GATEWAY not in hostname:
            print(
                f"[ERROR] ai_worker is only reachable from {AI_WORKER_GATEWAY}. "
                f"Current host: {hostname}"
            )
            sys.exit(1)

    global DRY_RUN
    DRY_RUN = args.dry_run

    remote = args.remote
    source_base = SERVER_PATHS.get(remote, DEFAULT_SERVER_PATH)

    print(f"Remote:   {remote}")
    print(f"Datasets: {', '.join(DATASETS)}")
    if DRY_RUN:
        print("[DRY RUN]")
    print()

    ans = input("Pull all datasets? [y/N] ").strip().lower()
    if ans not in {"y", "yes"}:
        print("Aborted.")
        return

    print()
    for dataset in DATASETS:
        sync_dataset(remote, source_base, dataset)

    print("Done.")


if __name__ == "__main__":
    main()
