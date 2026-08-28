_base_ = ["./minddrive_qwen2_05B_infer.py"]

# Inference overlay for the counterfactual head trained with
# minddrive_qwen2_05b_train_cf_meta_aug.py.
#
# Keep architecture-defining options aligned with training, especially
# use_ego_meta_embedding=True. Candidate augmentation is a training-only
# regularizer, so it is disabled here.
model = dict(
    counterfactual_head=dict(
        type="CounterfactualReasoningHead",
        embed_dims=896,
        hidden_dims=256,
        future_steps=6,
        relevance_threshold=0.5,
        agent_score_threshold=0.25,
        valid_agent_labels=(0, 1, 2, 3, 7, 8),
        rel_loss_weight=1.0,
        speed_loss_weight=1.0,
        path_loss_weight=1.0,
        real_loss_weight=0.0,
        save_counterfactual_scenes=True,
        use_ego_meta_embedding=True,
        train_candidate_augmentation=False,
        aug_num_candidates=6,
        aug_loss_weight=0.5,
        aug_response_loss_weight=0.25,
    )
)
