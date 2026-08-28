_base_ = ["./minddrive_qwen2_05b_train_cf_mini.py"]

# Counterfactual head training with explicit ego meta-action conditioning and
# candidate perturbation augmentation.  The augmentation recomputes geometric
# factual relevance for perturbed ego futures; it does not assume or introduce
# counterfactual ground-truth agent futures.

model = dict(
    counterfactual_train_only=True,
    counterfactual_head=dict(
        type="CounterfactualReasoningHead",
        embed_dims=896,
        hidden_dims=256,
        future_steps=6,
        relevance_threshold=0.5,
        valid_agent_labels=(0, 1, 2, 3, 7, 8),
        min_valid_future_steps=2,
        rel_loss_weight=1.0,
        speed_loss_weight=1.0,
        path_loss_weight=1.0,
        real_loss_weight=0.05,
        save_counterfactual_scenes=False,
        use_ego_meta_embedding=True,
        train_candidate_augmentation=False,
        aug_num_candidates=6,
        aug_loss_weight=0.5,
        aug_response_loss_weight=0.25,
    ),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        ann_file="data/infos/b2d_infos_train.pkl",
        map_file="data/infos/b2d_map_infos.pkl",
    ),
    val=dict(
        ann_file="data/infos/b2d_infos_val.pkl",
        map_file="data/infos/b2d_map_infos.pkl",
    ),
    test=dict(
        ann_file="data/infos/b2d_infos_val.pkl",
        map_file="data/infos/b2d_map_infos.pkl",
    ),
)

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=1.0e-4,
    betas=(0.9, 0.999),
    weight_decay=1.0e-4,
)
runner = dict(type="IterBasedRunner", max_iters=20000)
checkpoint_config = dict(interval=5000, max_keep_ckpts=3)
evaluation = dict(interval=1000000)
log_config = dict(
      interval=50,
      hooks=[
          dict(type="TextLoggerHook"),
          dict(
              type="WandbLoggerHook",
              project="minddrive-counterfactual",
              name="cf_valid_agents_4gpu_bs1_10k",
              tags=["counterfactual", "valid-agent", "4gpu"],
          ),
      ],
  )
optimizer_config = dict(
      type="Fp16OptimizerHook",
      loss_scale="dynamic",
      grad_clip=dict(
          max_norm=35,
          norm_type=2,
      ),
  )
lr_config = dict(
    _delete_=True,
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr_ratio=0.01,
)

find_unused_parameters = True
load_from = None
resume_from = None
