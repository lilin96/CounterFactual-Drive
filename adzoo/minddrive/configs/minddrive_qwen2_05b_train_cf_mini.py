_base_ = ["./minddrive_qwen2_05b_train_stage3.py"]

# Mini smoke training for the counterfactual head only.
# It reuses frozen MindDrive map/object scene tokens and trains only loss_cf_*.

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    "car",
    "van",
    "truck",
    "bicycle",
    "traffic_sign",
    "traffic_cone",
    "traffic_light",
    "pedestrian",
    "others",
]
ida_aug_conf = dict(
    resize_lim=(0.37, 0.45),
    final_dim=(320, 640),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=False,
)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
collect_keys = ["lidar2img", "cam_intrinsic", "timestamp", "ego_pose", "ego_pose_inv", "command"]

model = dict(
    counterfactual_train_only=True,
    counterfactual_head=dict(
        type="CounterfactualReasoningHead",
        embed_dims=896,
        hidden_dims=256,
        future_steps=6,
        relevance_threshold=0.5,
        rel_loss_weight=1.0,
        speed_loss_weight=1.0,
        path_loss_weight=1.0,
        real_loss_weight=0.05,
    ),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        ann_file="data/infos-mini/b2d_infos_val_existing.pkl",
        map_file="data/infos-mini/b2d_map_infos.pkl",
        pipeline=[
            dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True),
            dict(type="PhotoMetricDistortionMultiViewImage"),
            dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=True, with_light_state=True),
            dict(type="VADObjectRangeFilter", point_cloud_range=point_cloud_range),
            dict(type="VADObjectNameFilter", classes=class_names),
            dict(type="ResizeCropFlipRotImage", data_aug_conf=ida_aug_conf, training=True),
            dict(type="ResizeMultiview3D", img_scale=(640, 640), keep_ratio=False, multiscale_mode="value"),
            dict(type="PadMultiViewImage", size_divisor=32),
            dict(type="NormalizeMultiviewImage", **img_norm_cfg),
            dict(type="PETRFormatBundle3D", class_names=class_names, collect_keys=collect_keys),
            dict(
                type="CustomCollect3D",
                keys=[
                    "gt_bboxes_3d",
                    "gt_labels_3d",
                    "img",
                    "ego_his_trajs",
                    "gt_attr_labels",
                    "ego_fut_trajs",
                    "ego_fut_masks",
                    "ego_fut_cmd",
                    "ego_lcf_feat",
                    "can_bus",
                    "traffic_state_mask",
                    "path_points_future",
                    "path_future_mask",
                    "traffic_state",
                ] + collect_keys,
            ),
        ],
    ),
    val=dict(
        ann_file="data/infos-mini/b2d_infos_val_existing.pkl",
        map_file="data/infos-mini/b2d_map_infos.pkl",
    ),
    test=dict(
        ann_file="data/infos-mini/b2d_infos_val_existing.pkl",
        map_file="data/infos-mini/b2d_map_infos.pkl",
    ),
)

optimizer = dict(type="AdamW", lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy="CosineAnnealing", warmup=None, min_lr_ratio=1e-2)

runner = dict(type="IterBasedRunner", max_iters=20)
checkpoint_config = dict(interval=10, max_keep_ckpts=2)
evaluation = dict(interval=1000000)
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            project="minddrive-counterfactual",
            name="cf_base_train",
            tags=["minddrive", "counterfactual", "base"],
        ),
    ],
)

find_unused_parameters=True
load_from=None
resume_from=None
