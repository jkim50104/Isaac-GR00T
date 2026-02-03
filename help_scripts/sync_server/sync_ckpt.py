#!/usr/bin/env python3
import subprocess
from pathlib import Path
import sys

REMOTE = "pearl"
BASE_LOCAL = Path("output") # Run from workspace
BASE_REMOTE = "/home/jokim/projects/Isaac-GR00T/output"

# IMPORTANT: your remote has ffw_* dirs. If you filter only ai_worker_*, you'll see nothing.
# None => include all. Or e.g. ("ffw_", "ai_worker_")
EXPERIMENT_PREFIXES: tuple[str, ...] | None = None

SHOW_OK = True
ENABLE_COPY = True  # master switch


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
    """
    List immediate child directory names under a remote path (portable).
    """
    entries = ssh_lines(f"ls -1 {path} 2>/dev/null || true")
    if not entries:
        return []

    # Filter to dirs via a remote loop (portable)
    cmd = (
        f"cd {path} 2>/dev/null || exit 0; "
        f"for x in *; do [ -d \"$x\" ] && echo \"$x\"; done"
    )
    dirs = ssh_lines(cmd)
    dirs.sort()
    return dirs


def list_remote_experiments() -> list[str]:
    exps = list_remote_dirs(BASE_REMOTE)
    exps = [e for e in exps if _prefix_ok(e)]
    exps.sort()
    return exps


def list_remote_hparams(exp: str) -> list[str]:
    return list_remote_dirs(f"{BASE_REMOTE}/{exp}")


def ckpt_sort_key(name: str):
    try:
        return int(name.split("-", 1)[1])
    except Exception:
        return name


def list_remote_ckpts(exp: str, hp: str) -> list[str]:
    # Only checkpoint-* directories
    entries = ssh_lines(f"ls -1 {BASE_REMOTE}/{exp}/{hp} 2>/dev/null || true")
    ckpts = [x for x in entries if x.startswith("checkpoint-")]
    ckpts.sort(key=ckpt_sort_key)
    return ckpts


def local_ckpt_exists(exp: str, hp: str, ckpt: str) -> bool:
    return (BASE_LOCAL / exp / hp / ckpt).is_dir()


def format_row(left: str, right: str, status: str, w: int) -> str:
    return f"  {left:<{w}}  {right:<{w}}  {status}"


def rsync_copy(exp: str, hp: str, ckpt: str):
    src = f"{REMOTE}:{BASE_REMOTE}/{exp}/{hp}/{ckpt}/"
    dst = str((BASE_LOCAL / exp / hp / ckpt).resolve()) + "/"
    (BASE_LOCAL / exp / hp).mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", "--info=progress2", src, dst])


# ---------------------------
# Interactive selection utils
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
    """
    Show numbered list and allow user to pick indices or all.
    Returns selected items (strings).
    """
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


def main():
    remote_exps = list_remote_experiments()

    print(f"[DEBUG] REMOTE={REMOTE}")
    print(f"[DEBUG] BASE_REMOTE={BASE_REMOTE}")
    print(f"[DEBUG] found remote experiments: {len(remote_exps)}")
    if remote_exps:
        print(f"[DEBUG] first few: {remote_exps[:5]}")

    # (exp, hp) -> [ckpt, ckpt, ...]
    to_copy_by_folder: dict[tuple[str, str], list[str]] = {}

    for exp in remote_exps:
        hps = list_remote_hparams(exp)
        if not hps:
            continue

        for hp in hps:
            ckpts = list_remote_ckpts(exp, hp)
            if not ckpts:
                continue

            w = max([len("(missing)"), *(len(x) for x in ckpts)] or [10])

            print(f"\n=== {exp}/{hp}/ ===")
            print(f"  {'LOCAL(A)':<{w}}  {'REMOTE(B)':<{w}}  STATUS")
            print(f"  {'-'*w}  {'-'*w}  ------")

            for ck in ckpts:
                if local_ckpt_exists(exp, hp, ck):
                    if SHOW_OK:
                        print(format_row(ck, ck, "OK", w))
                else:
                    print(format_row("(missing)", ck, "MISSING_ON_A", w))
                    key = (exp, hp)
                    to_copy_by_folder.setdefault(key, []).append(ck)

    if not ENABLE_COPY or not to_copy_by_folder:
        print("\nNo checkpoints need copying.")
        return

    # ---------------------------
    # Step 1: choose folders
    # ---------------------------
    folder_keys = sorted(to_copy_by_folder.keys())
    folder_labels = [f"{exp}/{hp}" for exp, hp in folder_keys]

    selected_folders = choose_items_menu(
        "COPY PREVIEW (Folders with missing checkpoints)",
        folder_labels,
    )
    if not selected_folders:
        print("\n❌ Aborted. No files were copied.")
        sys.exit(0)

    selected_folder_keys = [folder_keys[folder_labels.index(lbl)] for lbl in selected_folders]

    # ---------------------------
    # Step 2: per folder choose ckpts (all or indexed)
    # ---------------------------
    final_to_copy: list[tuple[str, str, str]] = []

    for (exp, hp) in selected_folder_keys:
        ckpts = sorted(to_copy_by_folder[(exp, hp)], key=ckpt_sort_key)
        folder_title = f"{exp}/{hp} (missing checkpoints)"

        print(f"\n=== {folder_title} ===")
        print("Options:")
        print("  [1] Download ALL missing checkpoints in this folder")
        print("  [2] Select checkpoints by index")
        print("  [3] Skip this folder")
        choice = input("> ").strip()

        if choice == "1":
            for ck in ckpts:
                final_to_copy.append((exp, hp, ck))
        elif choice == "2":
            picked = choose_items_menu(folder_title, ckpts)
            for ck in picked:
                final_to_copy.append((exp, hp, ck))
        else:
            print("Skipping.")
            continue

    if not final_to_copy:
        print("\nNo checkpoints selected. Nothing to copy.")
        return

    # ---------------------------
    # Final preview + confirm
    # ---------------------------
    print("\n=== FINAL COPY PREVIEW ===")
    print("The following checkpoints WILL BE COPIED from REMOTE(B) → LOCAL(A):\n")
    for exp, hp, ck in final_to_copy:
        print(f"  {exp}/{hp}/{ck}")

    if not confirm("Proceed with copying?"):
        print("\n❌ Aborted. No files were copied.")
        sys.exit(0)

    print("\n✅ Starting copy...\n")
    for exp, hp, ck in final_to_copy:
        rsync_copy(exp, hp, ck)

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()
