#!/usr/bin/env python3
import subprocess
from pathlib import Path
import sys

# ---------------------------
# Config
# ---------------------------
REMOTE = "neuron"  # pearl, turing, lunar, neuron

BASE_REMOTE_MAP = {
    "pearl":  "/home/jokim/projects/Isaac-GR00T/data/jkim50104",
    "turing": "/home/jokim/projects/Isaac-GR00T/data/jkim50104",
    "lunar":  "/home/robi/projects/Isaac-GR00T/data/jkim50104",
    "neuron": "/home01/hpc197a03/scratch/projects/Isaac-GR00T/data/jkim50104",
}

try:
    BASE_REMOTE = BASE_REMOTE_MAP[REMOTE]
except KeyError:
    raise ValueError(f"Unknown REMOTE '{REMOTE}'. Choose from {list(BASE_REMOTE_MAP)}")

print(BASE_REMOTE)


# Local base (A)
BASE_LOCAL = Path("data/jkim50104")

REQUIRED_SUBDIRS = ("data", "meta", "videos")

# Excludes only for meta sync
META_EXCLUDES = ("relative_stats.json", "stats.json", "modality.json")

# None => include all; or e.g. ("ffw_",)
EXPERIMENT_PREFIXES: tuple[str, ...] | None = None

SHOW_OK = True
ENABLE_COPY = True
DRY_RUN = False  # if True, rsync uses -n only


# ---------------------------
# tqdm fallback
# ---------------------------
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(it, **kwargs):
        return it


# ---------------------------
# Basic helpers
# ---------------------------
def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def _prefix_ok(name: str) -> bool:
    if EXPERIMENT_PREFIXES is None:
        return True
    return any(name.startswith(p) for p in EXPERIMENT_PREFIXES)


def ssh_lines(cmd: str) -> list[str]:
    """Run a remote command and return non-empty stripped lines. Print debug on failure."""
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
    """List immediate child directory names under a remote path (portable)."""
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


def list_local_dirs(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted([p.name for p in path.iterdir() if p.is_dir()])


# ---------------------------
# Health checks (trio present)
# ---------------------------
def local_exp_exists(exp: str) -> bool:
    return (BASE_LOCAL / exp).is_dir()


def local_subdirs_present(exp: str) -> set[str]:
    base = BASE_LOCAL / exp
    present: set[str] = set()
    for sd in REQUIRED_SUBDIRS:
        if (base / sd).is_dir():
            present.add(sd)
    return present


def remote_exp_exists(exp: str) -> bool:
    out = ssh_lines(f'[ -d "{BASE_REMOTE}/{exp}" ] && echo OK || echo NO')
    return bool(out) and out[-1] == "OK"


def remote_subdirs_present(exp: str) -> set[str]:
    present: set[str] = set()
    if not remote_exp_exists(exp):
        return present
    for sd in REQUIRED_SUBDIRS:
        out = ssh_lines(f'[ -d "{BASE_REMOTE}/{exp}/{sd}" ] && echo OK || echo NO')
        if out and out[-1] == "OK":
            present.add(sd)
    return present


# ---------------------------
# UI selection helpers
# ---------------------------
def _parse_index_set(s: str, n: int) -> set[int]:
    """
    Parse user input like:
      "1 2 5", "1,2,5", "1-3,7", "all"
    Returns 0-based indices.
    """
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
                lo_i = int(lo)
                hi_i = int(hi)
            except ValueError:
                continue
            lo0 = lo_i - 1
            hi0 = hi_i - 1
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


def choose_items_menu(title: str, items: list[str]) -> list[str]:
    print(f"\n=== {title} ===")
    for i, it in enumerate(items, 1):
        print(f"  [{i:>2}] {it}")

    print("\nSelect by index (e.g. '1 3 5' or '1-3,7') or 'all'.")
    print("Type empty / anything invalid to abort.")
    ans = input("> ").strip()
    idxs = _parse_index_set(ans, len(items))
    if not idxs:
        return []
    return [items[i] for i in sorted(idxs)]


def confirm(prompt: str) -> bool:
    print(f"\n{prompt}")
    print("Type 'yes' to proceed, anything else to abort:")
    return input("> ").strip().lower() == "yes"


# ---------------------------
# rsync helpers (missing-only + meta excludes)
# ---------------------------
def _rsync_exclude_args(subdir: str) -> list[str]:
    if subdir != "meta":
        return []
    args: list[str] = []
    for f in META_EXCLUDES:
        args += ["--exclude", f]
    return args


def _rsync_missing_count(src: str, dst: str, subdir: str) -> int:
    """
    Count how many files would be copied from src -> dst using:
      rsync -a --ignore-existing -n --out-format=%n (+ excludes for meta)
    """
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


def _ensure_remote_dir(path: str) -> None:
    ssh_lines(f'mkdir -p "{path}"')


def _ensure_local_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rsync_pull(exp: str, sd: str) -> None:
    """remote(B) -> local(A), missing-only"""
    src = f"{REMOTE}:{BASE_REMOTE}/{exp}/{sd}/"
    dst_path = (BASE_LOCAL / exp / sd).resolve()
    _ensure_local_dir(dst_path)

    cmd = ["rsync", "-a", "--ignore-existing", "--info=progress2", "--partial"]
    cmd += _rsync_exclude_args(sd)
    if DRY_RUN:
        cmd.append("-n")
    cmd += [src, str(dst_path) + "/"]
    run(cmd)


def rsync_push(exp: str, sd: str) -> None:
    """local(A) -> remote(B), missing-only"""
    src_path = (BASE_LOCAL / exp / sd).resolve()
    dst = f"{REMOTE}:{BASE_REMOTE}/{exp}/{sd}/"
    _ensure_remote_dir(f"{BASE_REMOTE}/{exp}/{sd}")

    cmd = ["rsync", "-a", "--ignore-existing", "--info=progress2", "--partial"]
    cmd += _rsync_exclude_args(sd)
    if DRY_RUN:
        cmd.append("-n")
    cmd += [str(src_path) + "/", dst]
    run(cmd)


# ---------------------------
# Main
# ---------------------------
def main():
    print(f"[DEBUG] REMOTE={REMOTE}")
    print(f"[DEBUG] BASE_REMOTE={BASE_REMOTE}")
    print(f"[DEBUG] BASE_LOCAL={BASE_LOCAL.resolve()}")
    print(f"[DEBUG] REQUIRED_SUBDIRS={REQUIRED_SUBDIRS}")
    print(f"[DEBUG] META_EXCLUDES={META_EXCLUDES}")

    remote_exps = [e for e in list_remote_dirs(BASE_REMOTE) if _prefix_ok(e)]
    local_exps = [e for e in list_local_dirs(BASE_LOCAL) if _prefix_ok(e)]
    all_exps = sorted(set(remote_exps) | set(local_exps))

    if not all_exps:
        print("\nNo experiments found on either side.")
        return

    corrupt: list[str] = []
    eligible_rows: list[dict] = []

    # Scan & classify (with progress)
    for exp in tqdm(all_exps, desc="Scanning experiments", unit="exp"):
        l_exist = local_exp_exists(exp)
        r_exist = remote_exp_exists(exp)

        l_present = local_subdirs_present(exp) if l_exist else set()
        r_present = remote_subdirs_present(exp) if r_exist else set()

        l_corrupt = l_exist and (l_present != set(REQUIRED_SUBDIRS))
        r_corrupt = r_exist and (r_present != set(REQUIRED_SUBDIRS))

        # corruption: exists but missing any of trio -> warn, skip
        if l_corrupt or r_corrupt:
            msg_parts = []
            if l_corrupt:
                missing = sorted(set(REQUIRED_SUBDIRS) - l_present)
                msg_parts.append(f"LOCAL missing {missing}")
            if r_corrupt:
                missing = sorted(set(REQUIRED_SUBDIRS) - r_present)
                msg_parts.append(f"REMOTE missing {missing}")
            corrupt.append(f"{exp}  ({' | '.join(msg_parts)})")
            continue

        # Eligible
        eligible_rows.append(
            {"exp": exp, "local_exist": l_exist, "remote_exist": r_exist}
        )

    # Show corruption warnings
    if corrupt:
        print("\n⚠️  CORRUPTION WARNINGS (will NOT sync these):")
        for line in corrupt:
            print(f"  - {line}")

    if not eligible_rows:
        print("\nNo eligible experiments to sync (everything is corrupt or missing).")
        return

    # Build preview lines with pull/push counts (with progress)
    preview_items: list[str] = []
    exp_lookup: dict[str, str] = {}

    for row in tqdm(eligible_rows, desc="Computing sync deltas", unit="exp"):
        exp = row["exp"]
        l_exist = row["local_exist"]
        r_exist = row["remote_exist"]

        pull_n = 0
        push_n = 0

        # Ensure both sides directories exist for dry-run counting
        if r_exist:
            for sd in REQUIRED_SUBDIRS:
                src = f"{REMOTE}:{BASE_REMOTE}/{exp}/{sd}/"
                dst = str((BASE_LOCAL / exp / sd).resolve()) + "/"
                _ensure_local_dir((BASE_LOCAL / exp / sd).resolve())
                pull_n += _rsync_missing_count(src, dst, sd)

        if l_exist:
            for sd in REQUIRED_SUBDIRS:
                src = str((BASE_LOCAL / exp / sd).resolve()) + "/"
                dst = f"{REMOTE}:{BASE_REMOTE}/{exp}/{sd}/"
                _ensure_remote_dir(f"{BASE_REMOTE}/{exp}/{sd}")
                push_n += _rsync_missing_count(src, dst, sd)

        local_s = "OK" if l_exist else "ABSENT"
        remote_s = "OK" if r_exist else "ABSENT"
        tag = "IN_SYNC" if (pull_n == 0 and push_n == 0) else f"PULL:{pull_n} PUSH:{push_n}"

        line = f"{exp:<45}  LOCAL:{local_s:<7}  REMOTE:{remote_s:<7}  {tag}"
        preview_items.append(line)
        exp_lookup[line] = exp

    # Choose experiments (same style as your ckpt script)
    print("\n=== SYNC PREVIEW (Eligible experiments) ===")
    for i, line in enumerate(preview_items, 1):
        print(f"  [{i:>2}] {line}")

    print("\nSelect by index (e.g. '1 3 5' or '1-3,7') or 'all'.")
    print("Type empty / anything invalid to abort.")
    ans = input("> ").strip()
    idxs = _parse_index_set(ans, len(preview_items))
    if not idxs:
        print("\n❌ Aborted. No sync performed.")
        sys.exit(0)

    selected_lines = [preview_items[i] for i in sorted(idxs)]
    selected_exps = [exp_lookup[line] for line in selected_lines]

    # Step 2: per experiment choose subdirs (all or select)
    plan: list[tuple[str, str]] = []
    for exp in selected_exps:
        print(f"\n=== {exp} ===")
        print("Options:")
        print("  [1] Sync ALL (data/meta/videos) bidirectionally (missing-only)")
        print("  [2] Select subdirs by index")
        print("  [3] Skip this experiment")
        choice = input("> ").strip()

        if choice == "1":
            for sd in REQUIRED_SUBDIRS:
                plan.append((exp, sd))
        elif choice == "2":
            picked = choose_items_menu(f"{exp} (select subdirs)", list(REQUIRED_SUBDIRS))
            for sd in picked:
                plan.append((exp, sd))
        else:
            print("Skipping.")
            continue

    if not plan:
        print("\nNo items selected. Nothing to sync.")
        return

    # Final preview (per exp/subdir pull/push counts) with tqdm
    final_lines: list[str] = []
    for exp, sd in tqdm(plan, desc="Finalizing preview", unit="dir"):
        src_pull = f"{REMOTE}:{BASE_REMOTE}/{exp}/{sd}/"
        dst_pull = str((BASE_LOCAL / exp / sd).resolve()) + "/"
        _ensure_local_dir((BASE_LOCAL / exp / sd).resolve())
        pull_n = _rsync_missing_count(src_pull, dst_pull, sd)

        src_push = str((BASE_LOCAL / exp / sd).resolve()) + "/"
        dst_push = f"{REMOTE}:{BASE_REMOTE}/{exp}/{sd}/"
        _ensure_remote_dir(f"{BASE_REMOTE}/{exp}/{sd}")
        push_n = _rsync_missing_count(src_push, dst_push, sd)

        final_lines.append(f"{exp}/{sd:<6}  PULL:{pull_n:<6}  PUSH:{push_n:<6}")

    print("\n=== FINAL BIDIRECTIONAL SYNC PREVIEW (missing-only) ===")
    print("(meta excludes: relative_stats.json, stats.json)")
    for line in final_lines:
        print(f"  {line}")

    if not ENABLE_COPY:
        print("\nENABLE_COPY is False; exiting without syncing.")
        return

    if not confirm("Proceed with syncing these?"):
        print("\n❌ Aborted. No sync performed.")
        sys.exit(0)

    print("\n✅ Starting sync...\n")
    for exp, sd in tqdm(plan, desc="Syncing", unit="dir"):
        # pull then push, both missing-only; excludes apply for meta
        rsync_pull(exp, sd)
        rsync_push(exp, sd)

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()
