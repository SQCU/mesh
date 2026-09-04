MATRIX_FUSION_ARMS = (
    "matrix_fusion",
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
OPTIMIZATION_ARMS = ("matrix_fusion", "ffn", "linear")
STUDY_ARMS = ("matrix_fusion", "initial_policy", "ffn", "linear", "default")

def architecture_arm(arm):
    return "matrix_fusion" if arm in MATRIX_FUSION_ARMS else arm

def is_matrix_fusion_arm(arm):
    return arm in MATRIX_FUSION_ARMS

__all__ = ["MATRIX_FUSION_ARMS", "MATRIX_FUSION_INTERVENTION_ARMS", "PARAMETERIZED_ARMS", "OPTIMIZATION_ARMS", "STUDY_ARMS", "architecture_arm", "is_matrix_fusion_arm"]
