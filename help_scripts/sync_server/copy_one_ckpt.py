#!/usr/bin/env python3
import subprocess
from pathlib import Path

# ===================== EDIT THESE =====================
WORKER_DIR = "ffw_sg2_rev1_pick_bowl/G4_B512_REL_AO_WR"
CHECKPOINT = "checkpoint-15000"
# ======================================================

REMOTE = "pearl"
BASE_REMOTE = "/home/jokim/projects/Isaac-GR00T/output"

# If you run from workspace/output, this is perfect.
# Otherwise change to Path("output") if needed.
BASE_LOCAL = Path(".")

def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    src = f"{REMOTE}:{BASE_REMOTE}/{WORKER_DIR}/{CHECKPOINT}/"
    dst = (BASE_LOCAL / WORKER_DIR / CHECKPOINT).resolve()

    print("\n=== SINGLE CHECKPOINT COPY ===")
    print(f"REMOTE: {src}")
    print(f"LOCAL : {dst}/\n")

    # make local dir
    dst.parent.mkdir(parents=True, exist_ok=True)

    run([
        "rsync",
        "-a",
        "--info=progress2",
        "--partial",   # resume support
        src,
        str(dst) + "/",
    ])

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
