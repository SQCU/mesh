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

from .losses import DEFAULT_LOSS_CONFIG

POLICY_ACTOR_WEIGHT = DEFAULT_LOSS_CONFIG.actor_weight
POLICY_MOE_BALANCE_WEIGHT = DEFAULT_LOSS_CONFIG.moe_balance_weight
POLICY_RATIO_CLIP = DEFAULT_LOSS_CONFIG.ratio_clip
POLICY_ROLLOUT_IS_THRESHOLD = DEFAULT_LOSS_CONFIG.rollout_is_threshold
POLICY_LOG_RATIO_BOUND = DEFAULT_LOSS_CONFIG.log_ratio_bound
