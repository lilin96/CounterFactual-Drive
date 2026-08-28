_base_ = ["./minddrive_qwen2_05B_infer_counterfactual_meta_aug.py"]

# A2 paper ablation: keep the learned graph/relevance outputs, but evaluate
# every ego candidate against the same original MindDrive agent futures.
model = dict(
    counterfactual_head=dict(
        use_candidate_conditioned_agent_response=False,
    )
)
