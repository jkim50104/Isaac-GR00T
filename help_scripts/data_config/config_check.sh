#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path/to/modality_config.py> [--strict]"
  echo "Example: $0 ./help_scripts/data_config/ai_worker_config.py --strict"
  exit 1
fi

CFG_PATH="$1"
STRICT=0
if [[ "${2:-}" == "--strict" ]]; then
  STRICT=1
fi

echo "=========== CONFIG CHECK =========="
echo "Config file: ${CFG_PATH}"
echo "Env (if used by config):"
echo "  GR00T_ARM_ONLY   = ${GR00T_ARM_ONLY:-<unset>}"
echo "  GR00T_USE_WRIST_VIEW  = ${GR00T_USE_WRIST_VIEW:-<unset>}"
echo "  GR00T_ACTION_REP = ${GR00T_ACTION_REP:-<unset>}"
echo "==================================="

python - "$CFG_PATH" "$STRICT" <<'PY'
import importlib.util
import sys

cfg_path = sys.argv[1]
strict = int(sys.argv[2])

def enum_name(x):
    return getattr(x, "name", str(x))

spec = importlib.util.spec_from_file_location("cfg_mod", cfg_path)
if spec is None or spec.loader is None:
    print(f"ERROR: Could not load config from: {cfg_path}", file=sys.stderr)
    sys.exit(2)

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

if not hasattr(m, "ai_worker"):
    print("ERROR: Config module does not define `ai_worker` dict.", file=sys.stderr)
    sys.exit(2)

ai = m.ai_worker

def get_mod(name):
    if name not in ai:
        print(f"ERROR: Missing modality '{name}' in ai_worker.", file=sys.stderr)
        sys.exit(2)
    return ai[name]

video = get_mod("video")
state = get_mod("state")
action = get_mod("action")
lang  = get_mod("language")

print("\n=========== RESOLVED MODALITY CONFIG ===========")
print("[video]")
print("  delta_indices:", list(getattr(video, "delta_indices", [])))
print("  keys        :", list(getattr(video, "modality_keys", [])))

print("\n[state]")
print("  delta_indices:", list(getattr(state, "delta_indices", [])))
print("  keys        :", list(getattr(state, "modality_keys", [])))

print("\n[action]")
delta = list(getattr(action, "delta_indices", []))
keys  = list(getattr(action, "modality_keys", []))
acs   = list(getattr(action, "action_configs", []) or [])

if delta:
    print(f"  delta_indices: {min(delta)}..{max(delta)} (len={len(delta)})")
else:
    print("  delta_indices: []")

print("  keys:", keys)
print("  configs (key -> rep/type/format):")

if len(acs) != len(keys):
    msg = f"#action_configs ({len(acs)}) != #action_keys ({len(keys)})"
    if strict:
        print("ERROR:", msg, file=sys.stderr)
        sys.exit(3)
    else:
        print("  !! WARNING:", msg)

for i, k in enumerate(keys):
    if i >= len(acs):
        line = f"{i:02d} {k}: MISSING ActionConfig"
        if strict:
            print("ERROR:", line, file=sys.stderr)
            sys.exit(3)
        print(" ", line)
        continue
    ac = acs[i]
    print(f"  {i:02d} {k}: rep={enum_name(ac.rep)} type={enum_name(ac.type)} format={enum_name(ac.format)}")

print("\n[language]")
print("  delta_indices:", list(getattr(lang, "delta_indices", [])))
print("  keys        :", list(getattr(lang, "modality_keys", [])))

print("================================================\n")
PY
