MATRIX_FUSION_ARMS = (
    "matrix_fusion",
    "terminal_win",
    "initial_policy",
    "participant_fusion_ablated",
    "residual_fusion_ablated",
)
MATRIX_FUSION_INTERVENTION_ARMS = (
    "matrix_fusion",
    "participant_fusion_ablated",
    "residual_fusion_ablated",
)
PARAMETERIZED_ARMS = (*MATRIX_FUSION_ARMS, "ffn", "linear")
OPTIMIZATION_ARMS = ("matrix_fusion", "terminal_win", "ffn", "linear")
STUDY_ARMS = ("matrix_fusion", "terminal_win", "initial_policy", "ffn", "linear", "default")
JOINT_TRAINING_ARMS = ("matrix_fusion", "terminal_win")

def checkpoint_path(path, arm):
    stem, extension = path.rsplit(".", 1)
    return f"{stem}.{arm}.{extension}"

def architecture_arm(arm):
    return "matrix_fusion" if arm in MATRIX_FUSION_ARMS else arm

def is_matrix_fusion_arm(arm):
    return arm in MATRIX_FUSION_ARMS

__all__ = ["MATRIX_FUSION_ARMS", "MATRIX_FUSION_INTERVENTION_ARMS", "PARAMETERIZED_ARMS", "OPTIMIZATION_ARMS", "STUDY_ARMS", "architecture_arm", "is_matrix_fusion_arm"]

POLICY_ACTOR_WEIGHT = 4.0
POLICY_MOE_BALANCE_WEIGHT = 0.01
POLICY_RATIO_CLIP = 0.2
POLICY_ROLLOUT_IS_THRESHOLD = 2.0
POLICY_LOG_RATIO_BOUND = 20.0
