_base_ = ["./minddrive_qwen25_3B_infer.py"]

# Counterfactual reasoning is disabled in existing configs by omission.
# Use this overlay to enable the lightweight MindDrive-integrated head while
# preserving the original MindDrive ego_fut_preds output.
model = dict(
    counterfactual_head=dict(
        type="CounterfactualReasoningHead",
        embed_dims=2048,
        hidden_dims=256,
        future_steps=6,
        relevance_threshold=0.5,
        rel_loss_weight=1.0,
        speed_loss_weight=1.0,
        path_loss_weight=1.0,
        real_loss_weight=0.0,
    )
)

