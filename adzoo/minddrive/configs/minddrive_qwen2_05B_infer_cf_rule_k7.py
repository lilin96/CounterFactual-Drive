_base_ = ["./minddrive_qwen2_05B_infer_counterfactual_meta_aug.py"]

# A3 paper ablation: seven deterministic rule candidates (one per speed meta
# action) with a fixed straight path.  Do not prepend an eighth base candidate.
model = dict(
    counterfactual_head=dict(
        rule_candidate_mode="speed_only",
        rule_fixed_path_idx=1,
    )
)
