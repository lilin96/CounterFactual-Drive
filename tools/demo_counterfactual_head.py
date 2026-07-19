"""Smoke test for CounterfactualReasoningHead.forward_infer with fake tensors."""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

try:
    from mmcv.models.counterfactual import CounterfactualReasoningHead
except ModuleNotFoundError:
    class _Registry:
        def register_module(self):
            def wrap(cls):
                return cls
            return wrap

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mmcv_stub = types.ModuleType("mmcv")
    models_stub = types.ModuleType("mmcv.models")
    builder_stub = types.ModuleType("mmcv.models.builder")
    cf_stub = types.ModuleType("mmcv.models.counterfactual")
    builder_stub.HEADS = _Registry()
    cf_stub.__path__ = [os.path.join(root, "mmcv", "models", "counterfactual")]
    sys.modules.setdefault("mmcv", mmcv_stub)
    sys.modules.setdefault("mmcv.models", models_stub)
    sys.modules.setdefault("mmcv.models.builder", builder_stub)
    sys.modules.setdefault("mmcv.models.counterfactual", cf_stub)
    from mmcv.models.counterfactual.counterfactual_head import CounterfactualReasoningHead


def main():
    torch.manual_seed(0)
    head = CounterfactualReasoningHead(embed_dims=32, hidden_dims=16, future_steps=6)
    scene_tokens = torch.randn(1, 12, 32)
    bbox_result = dict(
        boxes_3d=torch.randn(4, 7),
        scores_3d=torch.rand(4),
        labels_3d=torch.randint(0, 9, (4,)),
        trajs_3d=torch.randn(4, 6, 2).cumsum(dim=-2),
    )
    output = head.forward_infer(scene_tokens, bbox_result, lane_results=[{}])
    print("selected_ego_future", tuple(output["selected_ego_future"].shape))
    print("selected_meta_action", output["selected_meta_action"])
    print("risk_scores", sorted(output["risk_scores"].keys()))
    print("num_counterfactual_scenes", len(output["counterfactual_scenes"]))


if __name__ == "__main__":
    main()
