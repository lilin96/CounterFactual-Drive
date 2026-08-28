# Copyright (c) OpenMMLab. All rights reserved.
"""Weights & Biases logger hook for this local MMCV fork."""

from mmcv.utils import master_only

from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):
    """Log runner metrics to Weights & Biases.

    Args:
        project (str): WandB project name.
        name (str | None): WandB run name.
        entity (str | None): WandB entity/team.
        tags (list[str] | None): Optional run tags.
        interval (int): Logging interval.
        log_config (bool): Whether to attach the full MMCV config.
    """

    def __init__(
        self,
        project,
        name=None,
        entity=None,
        tags=None,
        interval=10,
        ignore_last=True,
        reset_flag=False,
        by_epoch=True,
        log_config=True,
    ):
        super(WandbLoggerHook, self).__init__(interval, ignore_last, reset_flag, by_epoch)
        self.project = project
        self.name = name
        self.entity = entity
        self.tags = tags
        self.log_config = log_config
        self.wandb = None

    @master_only
    def before_run(self, runner):
        super(WandbLoggerHook, self).before_run(runner)
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("Please install wandb before using WandbLoggerHook.") from exc

        self.wandb = wandb
        init_kwargs = dict(project=self.project, name=self.name, entity=self.entity, tags=self.tags)
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        if self.log_config and hasattr(runner, "meta") and runner.meta is not None:
            config = runner.meta.get("config", None)
            # wandb treats string config values as a path to a config file.
            # MindDrive stores cfg.pretty_text in runner.meta["config"], so only
            # pass structured configs here.
            if isinstance(config, dict):
                init_kwargs["config"] = config
        self.wandb.init(**init_kwargs)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=False)
        step = self.get_iter(runner)
        self.wandb.log(tags, step=step)

    @master_only
    def after_run(self, runner):
        if self.wandb is not None and self.wandb.run is not None:
            self.wandb.finish()
