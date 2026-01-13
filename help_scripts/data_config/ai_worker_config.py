from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat

ai_worker = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view",
                    #    "left_wrist_view",
                    #    "right_wrist_view",
                       ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
            "head",
            "lift",
            "base",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 32)),
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
            "head",
            "lift",
            "base",
        ],
        action_configs=[
            # left_arm
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # G1 hand is controlled by binary signals like a gripper
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # head
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # lift
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # base
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(ai_worker, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)