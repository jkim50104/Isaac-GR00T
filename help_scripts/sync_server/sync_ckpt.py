#!/usr/bin/env python3
import socket
import subprocess
import shlex
from pathlib import Path
import sys

# Machines allowed to RUN this script (local workstations).
# Training servers are in REMOTE_SERVERS — running from one of those is blocked.
LOCAL_HOSTS: tuple[str, ...] = ("lunar",)

REMOTE_SERVERS: dict[str, str] = {
    "turing": "/home/jokim/projects/Isaac-GR00T/output",
    "rosen":  "/home/jokim/project/Isaac-GR00T/output",
    "pearl":  "/home/jokim/projects/Isaac-GR00T/output",
}

BASE_LOCAL = Path("output")  # Run from workspace root
REMOTE: str = ""             # SSH host alias, set by main()
BASE_REMOTE: str = ""        # Remote output dir, set by main()

# None => include all legacy experiments. Or e.g. ("ffw_", "ai_worker_")
EXPERIMENT_PREFIXES: tuple[str, ...] | None = None

SHOW_OK = True
ENABLE_COPY = True    # master switch for copy
ENABLE_DELETE = True  # master switch for remote deletion

# Files that must exist for a checkpoint to be considered complete.
_REQUIRED_CKPT_FILES = [
    "config.json",
    "training_args.bin",
    "model.safetensors.index.json",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def _prefix_ok(name: str) -> bool:
    if EXPERIMENT_PREFIXES is None:
        return True
    return any(name.startswith(p) for p in EXPERIMENT_PREFIXES)


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


def ckpt_sort_key(name: str):
    try:
        return int(name.split("-", 1)[1])
    except Exception:
        return name


# ---------------------------------------------------------------------------
# Remote scan — single SSH call via find
# ---------------------------------------------------------------------------

def scan_remote() -> dict[tuple[str, str], list[str]]:
    """
    One SSH round-trip: find all checkpoint-* dirs under BASE_REMOTE.
    Returns {(exp, hp): [ckpt, ...]} sorted by checkpoint step.

    Handles both:
      New:    v{ver}/{dataset}/{hparams}/checkpoint-N  (depth 4)
      Legacy: {dataset}/{hparams}/checkpoint-N         (depth 3)
    """
    cmd = (
        f"find -H {BASE_REMOTE} -mindepth 3 -maxdepth 4 -name 'checkpoint-*' -type d "
        f"2>/dev/null | sort"
    )
    lines = ssh_lines(cmd)

    prefix = BASE_REMOTE.rstrip("/") + "/"
    result: dict[tuple[str, str], list[str]] = {}

    for line in lines:
        if not line.startswith(prefix):
            continue
        parts = line[len(prefix):].split("/")

        if len(parts) == 4 and parts[0].startswith("v"):
            exp, hp, ckpt = f"{parts[0]}/{parts[1]}", parts[2], parts[3]
        elif len(parts) == 3:
            exp, hp, ckpt = parts[0], parts[1], parts[2]
            if not _prefix_ok(exp):
                continue
        else:
            continue

        if not ckpt.startswith("checkpoint-"):
            continue

        result.setdefault((exp, hp), []).append(ckpt)

    for key in result:
        result[key].sort(key=ckpt_sort_key)

    return result


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

def rsync_copy(exp: str, hp: str, ckpt: str) -> None:
    src = f"{REMOTE}:{BASE_REMOTE}/{exp}/{hp}/{ckpt}/"
    dst = str((BASE_LOCAL / exp / hp / ckpt).resolve()) + "/"
    (BASE_LOCAL / exp / hp).mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", "--info=progress2", src, dst])


# ---------------------------------------------------------------------------
# Delete + verification
# ---------------------------------------------------------------------------

def ssh_delete_ckpt(exp: str, hp: str, ckpt: str) -> None:
    run(["ssh", REMOTE, f"rm -rf {BASE_REMOTE}/{exp}/{hp}/{ckpt}"])


def ssh_delete_folder_if_no_ckpts(exp: str, hp: str) -> None:
    folder = shlex.quote(f"{BASE_REMOTE}/{exp}/{hp}")
    cmd = (
        f"if [ -d {folder} ] && "
        f"""[ -z "$(find -H {folder} -mindepth 1 -maxdepth 1 """
        f"""-name 'checkpoint-*' -type d -print -quit)" ]; """
        f"then rm -rf {folder}; fi"
    )
    run(["ssh", REMOTE, cmd])


def _local_ckpt_bytes(exp: str, hp: str, ckpt: str) -> int:
    return sum(
        f.stat().st_size
        for f in (BASE_LOCAL / exp / hp / ckpt).rglob("*")
        if f.is_file()
    )


def _remote_ckpt_bytes_batch(keys: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], int]:
    """Single SSH du call for multiple checkpoint dirs."""
    if not keys:
        return {}
    paths = " ".join(f"{BASE_REMOTE}/{exp}/{hp}/{ck}" for exp, hp, ck in keys)
    lines = ssh_lines(f"du -sb {paths} 2>/dev/null")
    prefix = BASE_REMOTE.rstrip("/") + "/"
    sizes: dict[tuple[str, str, str], int] = {}
    for line in lines:
        tok = line.split(None, 1)
        if len(tok) != 2:
            continue
        try:
            size = int(tok[0])
        except ValueError:
            continue
        path = tok[1]
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix):].split("/")
        if len(parts) == 4 and parts[0].startswith("v"):
            key: tuple[str, str, str] = (f"{parts[0]}/{parts[1]}", parts[2], parts[3])
        elif len(parts) == 3:
            key = (parts[0], parts[1], parts[2])
        else:
            continue
        sizes[key] = size
    return sizes


def verify_batch(
    candidates: list[tuple[str, str, str]],
) -> dict[tuple[str, str, str], tuple[bool, str]]:
    """Verify all candidates with one SSH round-trip for sizes."""
    remote_sizes = _remote_ckpt_bytes_batch(candidates)
    results: dict[tuple[str, str, str], tuple[bool, str]] = {}

    for exp, hp, ck in candidates:
        ckpt_path = BASE_LOCAL / exp / hp / ck

        failed = False
        for fname in _REQUIRED_CKPT_FILES:
            if not (ckpt_path / fname).exists():
                results[(exp, hp, ck)] = (False, f"MISSING FILE: {fname}")
                failed = True
                break
        if failed:
            continue

        shards = list(ckpt_path.glob("model-*.safetensors"))
        if not shards and not (ckpt_path / "pytorch_model.bin").exists():
            results[(exp, hp, ck)] = (False, "MISSING FILE: model weights")
            continue

        local_sz = _local_ckpt_bytes(exp, hp, ck)
        remote_sz = remote_sizes.get((exp, hp, ck), 0)

        if remote_sz == 0:
            results[(exp, hp, ck)] = (False, "could not read remote size")
            continue

        diff_pct = abs(local_sz - remote_sz) / remote_sz
        gb = 1 << 30
        if diff_pct > 0.01:
            results[(exp, hp, ck)] = (
                False,
                f"SIZE MISMATCH: local={local_sz/gb:.2f}GB remote={remote_sz/gb:.2f}GB "
                f"({diff_pct:.1%} diff)",
            )
        else:
            results[(exp, hp, ck)] = (True, f"verified ({local_sz/gb:.2f} GB)")

    return results


# ---------------------------------------------------------------------------
# Interactive selection utils
# ---------------------------------------------------------------------------

def _parse_index_set(s: str, n: int) -> set[int]:
    s = s.strip().lower()
    if s in {"a", "all"}:
        return set(range(n))
    if not s:
        return set()
    s = s.replace(",", " ")
    out: set[int] = set()
    for p in s.split():
        if "-" in p:
            lo, hi = p.split("-", 1)
            try:
                lo0, hi0 = int(lo) - 1, int(hi) - 1
            except ValueError:
                continue
            if lo0 > hi0:
                lo0, hi0 = hi0, lo0
            out.update(i for i in range(lo0, hi0 + 1) if 0 <= i < n)
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
    print("\nSelect by index (e.g. '1 3 5' or '1-3,7') or 'all'. Empty to skip.")
    idxs = _parse_index_set(input("> ").strip(), len(items))
    return [items[i] for i in sorted(idxs)]


def confirm(prompt: str) -> bool:
    print(f"\n{prompt}")
    print("Type 'yes' to proceed, anything else to abort:")
    return input("> ").strip().lower() == "yes"


def format_row(left: str, right: str, status: str, w: int) -> str:
    return f"  {left:<{w}}  {right:<{w}}  {status}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    hostname = socket.gethostname()
    if any(h in hostname for h in REMOTE_SERVERS):
        print(f"[ERROR] Must run from a local machine. Current host: {hostname}")
        print(f"  Remote servers (not allowed): {', '.join(REMOTE_SERVERS)}")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: sync_ckpt.py <server>")
        print("\nAvailable remote servers:")
        for name, path in REMOTE_SERVERS.items():
            print(f"  {name:<12} {path}")
        sys.exit(0)

    server = sys.argv[1]
    if server not in REMOTE_SERVERS:
        print(f"Unknown server: {server}  (available: {', '.join(REMOTE_SERVERS)})")
        sys.exit(1)

    global REMOTE, BASE_REMOTE
    REMOTE = server
    BASE_REMOTE = REMOTE_SERVERS[server]

    print(f"[INFO] REMOTE={REMOTE}  BASE_REMOTE={BASE_REMOTE}")
    print("[INFO] Scanning remote checkpoints (single SSH call)...")

    remote_ckpts = scan_remote()  # {(exp, hp): [ckpt, ...]}

    print(f"[INFO] Found {len(remote_ckpts)} experiment/hparam folder(s).")

    to_copy_by_folder: dict[tuple[str, str], list[str]] = {}
    to_delete_by_folder: dict[tuple[str, str], list[str]] = {}

    for (exp, hp), ckpts in sorted(remote_ckpts.items()):
        w = max(len("(missing)"), *(len(c) for c in ckpts))
        print(f"\n=== {exp}/{hp}/ ===")
        print(f"  {'LOCAL(A)':<{w}}  {'REMOTE(B)':<{w}}  STATUS")
        print(f"  {'-'*w}  {'-'*w}  ------")
        for ck in ckpts:
            if (BASE_LOCAL / exp / hp / ck).is_dir():
                if SHOW_OK:
                    print(format_row(ck, ck, "OK (can delete remote)", w))
                to_delete_by_folder.setdefault((exp, hp), []).append(ck)
            else:
                print(format_row("(missing)", ck, "MISSING_ON_A", w))
                to_copy_by_folder.setdefault((exp, hp), []).append(ck)

    # ---------------------------
    # Copy phase
    # ---------------------------
    if ENABLE_COPY and to_copy_by_folder:
        folder_keys = sorted(to_copy_by_folder)
        folder_labels = [f"{e}/{h}" for e, h in folder_keys]

        selected = choose_items_menu("COPY — folders with missing checkpoints", folder_labels)

        final_to_copy: list[tuple[str, str, str]] = []
        for lbl in selected:
            exp, hp = folder_keys[folder_labels.index(lbl)]
            ckpts = to_copy_by_folder[(exp, hp)]
            print(f"\n=== {exp}/{hp} ===")
            print("  [1] Download ALL  [2] Select by index  [3] Skip")
            choice = input("> ").strip()
            if choice == "1":
                final_to_copy.extend((exp, hp, ck) for ck in ckpts)
            elif choice == "2":
                for ck in choose_items_menu(f"{exp}/{hp}", ckpts):
                    final_to_copy.append((exp, hp, ck))
            else:
                print("Skipping.")

        if final_to_copy:
            print("\n=== COPY PREVIEW — REMOTE(B) → LOCAL(A) ===")
            for exp, hp, ck in final_to_copy:
                print(f"  {exp}/{hp}/{ck}")
            if confirm("Proceed with copying?"):
                print()
                for exp, hp, ck in final_to_copy:
                    rsync_copy(exp, hp, ck)
                print("\n🎉 Copy done.")
            else:
                print("\n❌ Copy aborted.")
        else:
            print("\nNo checkpoints selected for copy.")
    else:
        print("\nNo checkpoints need copying.")

    # ---------------------------
    # Delete phase
    # ---------------------------
    if not ENABLE_DELETE or not to_delete_by_folder:
        return

    del_folder_keys = sorted(to_delete_by_folder)
    del_folder_labels = [f"{e}/{h}" for e, h in del_folder_keys]

    selected_del = choose_items_menu(
        "DELETE FROM REMOTE — checkpoints already synced locally",
        del_folder_labels,
    )
    if not selected_del:
        print("\nNo remote checkpoints deleted.")
        return

    final_to_delete: list[tuple[str, str, str]] = []
    for lbl in selected_del:
        exp, hp = del_folder_keys[del_folder_labels.index(lbl)]
        ckpts = to_delete_by_folder[(exp, hp)]
        for ck in choose_items_menu(f"{exp}/{hp} — pick checkpoints to delete", ckpts):
            final_to_delete.append((exp, hp, ck))

    if not final_to_delete:
        print("\nNo remote checkpoints selected.")
        return

    print("\n=== VERIFYING LOCAL CHECKPOINTS (single SSH call) ===")
    verification = verify_batch(final_to_delete)

    verified: list[tuple[str, str, str]] = []
    for exp, hp, ck in final_to_delete:
        ok, reason = verification.get((exp, hp, ck), (False, "not checked"))
        mark = "✅" if ok else "❌"
        print(f"  {mark} {exp}/{hp}/{ck}  — {reason}")
        if ok:
            verified.append((exp, hp, ck))

    blocked = [t for t in final_to_delete if t not in verified]
    if blocked:
        print(f"\n⚠️  {len(blocked)} checkpoint(s) failed verification — will NOT be deleted.")

    if not verified:
        print("\nNo checkpoints passed verification. Nothing deleted.")
        return

    print("\n=== DELETE PREVIEW — from REMOTE(B) ===")
    for exp, hp, ck in verified:
        print(f"  {REMOTE}:{BASE_REMOTE}/{exp}/{hp}/{ck}")

    if not confirm("Proceed with remote deletion?"):
        print("\n❌ Aborted. Nothing deleted.")
        return

    print()
    for exp, hp, ck in verified:
        ssh_delete_ckpt(exp, hp, ck)
    for exp, hp in sorted({(exp, hp) for exp, hp, _ in verified}):
        ssh_delete_folder_if_no_ckpts(exp, hp)

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()
