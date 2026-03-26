# help_scripts/data_config/ai_worker_config.py
import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ModalityConfig,
    ActionConfig,
    ActionRepresentation,
    ActionType,
    ActionFormat,
)

def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")

def _action_rep_from_env(name: str, default: str = "REL") -> ActionRepresentation:
    v = os.getenv(name, default).strip().upper()
    if v in ("REL", "RELATIVE"):
        return ActionRepresentation.RELATIVE
    if v in ("ABS", "ABSOLUTE"):
        return ActionRepresentation.ABSOLUTE
    raise ValueError(f"Unknown {name}={v} (use REL/ABS)")

# Read flags from environment (set these in your .sh before torchrun)
ARM_ONLY = _bool_env("GR00T_ARM_ONLY", default=False)
USE_WRIST_VIEW = _bool_env("GR00T_USE_WRIST_VIEW", default=True)

# IMPORTANT: this only affects left_arm and right_arm (grippers/head/lift/base remain ABSOLUTE)
ARM_ACTION_REP = _action_rep_from_env("GR00T_ACTION_REP", default="REL")

# Video modalities
video_keys = ["ego_view"]
if USE_WRIST_VIEW:
    video_keys += ["left_wrist_view", "right_wrist_view"]

# State/action modalities + action configs
if ARM_ONLY:
    state_keys = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    action_keys = state_keys[:]

    action_configs = [
        # left_arm (affected by GR00T_ACTION_REP)
        ActionConfig(rep=ARM_ACTION_REP, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        # left_gripper (always ABSOLUTE)
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        # right_arm (affected by GR00T_ACTION_REP)
        ActionConfig(rep=ARM_ACTION_REP, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        # right_gripper (always ABSOLUTE)
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
    ]
else:
    state_keys = [
        "left_arm",
        "left_gripper",
        "right_arm",
        "right_gripper",
        "head",
        "lift",
        "base",
    ]
    action_keys = state_keys[:]

    action_configs = [
        # left_arm (affected by GR00T_ACTION_REP)
        ActionConfig(rep=ARM_ACTION_REP, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        # left_gripper (always ABSOLUTE)
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        # right_arm (affected by GR00T_ACTION_REP)
        ActionConfig(rep=ARM_ACTION_REP, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        # right_gripper (always ABSOLUTE)
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        # head/lift/base (always ABSOLUTE regardless of GR00T_ACTION_REP)
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
    ]

ai_worker = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=video_keys,
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=state_keys,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 32)),
        modality_keys=action_keys,
        action_configs=action_configs,
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(ai_worker, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
