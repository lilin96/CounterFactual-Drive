_base_ = ["./minddrive_qwen2_05B_infer_counterfactual_meta_aug.py"]

# A1 paper ablation. This changes parameter shapes and therefore requires a
# separately trained checkpoint with use_ego_meta_embedding=False.
model = dict(
    counterfactual_head=dict(
        use_ego_meta_embedding=False,
    )
)
