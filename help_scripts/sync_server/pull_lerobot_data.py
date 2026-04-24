#!/usr/bin/env python3
"""Pull experiment datasets from a remote server to local (missing files only).

Two sources are supported:
  - server:    <remote>:projects/Isaac-GR00T/data/<dataset>/
  - ai_worker: <remote>:projects/physical_ai_tools/docker/huggingface/lerobot/<dataset>/
               (only available on servers that have a locally-connected ai_worker)
"""
import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
SERVERS = ["lunar", "turing", "pearl", "rosen"]
DATASETS = ["jkim50104", "ACS_ROBI"]

# Servers that have a locally-connected ai_worker (and thus a lerobot cache).
AI_WORKER_SERVERS = ["lunar"]

SOURCE_PATHS = {
    "server":    "projects/Isaac-GR00T/data",
    "ai_worker": "projects/physical_ai_tools/docker/huggingface/lerobot",
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-r", "--remote",
        choices=SERVERS,
        help=f"Remote server to pull from. Options: {', '.join(SERVERS)}",
    )
    parser.add_argument(
        "-s", "--source",
        choices=list(SOURCE_PATHS.keys()),
        default="server",
        help=(
            f"Which source to pull from (default: server). "
            f"'ai_worker' is only available on: {', '.join(AI_WORKER_SERVERS)}"
        ),
    )
    parser.add_argument(
        "-d", "--dataset",
        choices=DATASETS,
        default="jkim50104",
        help=f"Dataset directory to sync (default: jkim50104). Options: {', '.join(DATASETS)}",
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
        print("No remote server specified. Available servers:")
        for s in SERVERS:
            marker = "  (ai_worker ok)" if s in AI_WORKER_SERVERS else ""
            print(f"  - {s}{marker}")
        print(f"\nUsage: {sys.argv[0]} -r <server> [-s server|ai_worker] [-d <dataset>]")
        sys.exit(1)

    if args.source == "ai_worker" and args.remote not in AI_WORKER_SERVERS:
        print(
            f"[ERROR] source='ai_worker' is only available on servers with a "
            f"locally-connected ai_worker: {', '.join(AI_WORKER_SERVERS)}"
        )
        print(f"        You requested remote='{args.remote}'.")
        sys.exit(1)

    global DRY_RUN
    DRY_RUN = args.dry_run

    remote = args.remote
    remote_base = f"{SOURCE_PATHS[args.source]}/{args.dataset}"
    local_base = LOCAL_BASE / args.dataset

    print(f"Source:    {args.source}")
    print(f"Pull from: {remote}:{remote_base}")
    print(f"Local:     {local_base.resolve()}")
    print()

    remote_exps = sorted(list_remote_dirs(remote, remote_base))
    if not remote_exps:
        print("No experiments found on remote.")
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
            rsync_pull(remote, remote_base, local_base, exp, sd)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
