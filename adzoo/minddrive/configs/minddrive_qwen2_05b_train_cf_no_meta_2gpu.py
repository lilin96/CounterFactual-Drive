_base_ = ["./minddrive_qwen2_05b_train_cf_meta_aug.py"]

# A1 training ablation: remove only ego meta-action embedding.
# Two visible GPUs x 1 sample/GPU x 2 gradient accumulation = effective BS 4,
# matching the A0 four-GPU, one-sample-per-GPU training setup.
model = dict(
    counterfactual_head=dict(
        use_ego_meta_embedding=False,
    ),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
)

# 40k micro-iterations / accumulation 2 = 20k optimizer updates.  This also
# consumes 40k * 2 = 80k samples, equal to 20k * 4 for A0.
runner = dict(type="IterBasedRunner", max_iters=40000)
optimizer_config = dict(
    _delete_=True,
    type="GradientCumulativeFp16OptimizerHook",
    cumulative_iters=2,
    loss_scale="dynamic",
    grad_clip=dict(max_norm=35, norm_type=2),
)
lr_config = dict(
    _delete_=True,
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=0.1,
    min_lr_ratio=0.01,
)
checkpoint_config = dict(interval=10000, max_keep_ckpts=3)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            project="minddrive-counterfactual",
            name="cf_no_meta_2gpu_effbs4_20ksteps",
            tags=["counterfactual", "no-meta", "2gpu", "effective-bs4"],
        ),
    ],
)

load_from = None
resume_from = None
