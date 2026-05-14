#!/usr/bin/env python
import logging
import os

import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model

import train_net
from train_vigor_stage1 import freeze_swin_bottom_up
from vlpart import add_vlpart_config
from vlpart.config_object_prior import add_object_prior_config
from vlpart.data.dataset_mapper_object_prior import DatasetMapperWithImageObjectPrior

# Register experiment-only datasets / model classes without touching package __init__.py.
import vlpart.data.datasets.vigor_object_prior  # noqa: F401
import vlpart.modeling.meta_arch.vlm_rcnn_object_prior  # noqa: F401
import vlpart.modeling.roi_heads.object_prior_roi_heads  # noqa: F401


logger = logging.getLogger("detectron2")
_OBJECT_PRIOR_LOG_MODEL = None
_ORIGINAL_SWANLAB_LOG = train_net._log_swanlab


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _add_scalar_metric(metrics, name, value):
    if value is None:
        return
    metrics[name] = float(value.detach().float().cpu().item())


def _collect_object_prior_metrics(model):
    if model is None:
        return {}
    model = _unwrap_model(model)
    metrics = {}
    with torch.no_grad():
        fpn_alphas = getattr(model, "object_prior_fpn_alphas", None)
        if fpn_alphas is not None:
            fpn_alphas = fpn_alphas.detach().float().cpu()
            for idx, value in enumerate(fpn_alphas):
                metrics[f"object_prior/fpn_alpha_{idx}"] = float(value.item())
            metrics["object_prior/fpn_alpha_mean"] = float(fpn_alphas.mean().item())
            metrics["object_prior/fpn_alpha_min"] = float(fpn_alphas.min().item())
            metrics["object_prior/fpn_alpha_max"] = float(fpn_alphas.max().item())

        _add_scalar_metric(
            metrics,
            "object_prior/rpn_beta",
            getattr(model, "object_prior_rpn_beta", None),
        )
        roi_heads = getattr(model, "roi_heads", None)
        if roi_heads is not None:
            _add_scalar_metric(
                metrics,
                "object_prior/roi_beta",
                getattr(roi_heads, "object_prior_roi_beta", None),
            )
            _add_scalar_metric(
                metrics,
                "object_prior/mask_alpha",
                getattr(roi_heads, "object_prior_mask_alpha", None),
            )
    return metrics


def _log_swanlab_with_object_prior(swanlab_logger, metrics, step):
    if swanlab_logger is not None:
        metrics.update(_collect_object_prior_metrics(_OBJECT_PRIOR_LOG_MODEL))
    _ORIGINAL_SWANLAB_LOG(swanlab_logger, metrics, step)


def install_object_prior_swanlab_logging(model):
    global _OBJECT_PRIOR_LOG_MODEL
    _OBJECT_PRIOR_LOG_MODEL = model
    train_net._log_swanlab = _log_swanlab_with_object_prior


def setup(args):
    cfg = get_cfg()
    add_vlpart_config(cfg)
    add_object_prior_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if "/auto" in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("/auto", f"/{file_name}")
        logger.info("OUTPUT_DIR: %s", cfg.OUTPUT_DIR)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # Reuse train_net.do_train unchanged, but swap its mapper symbol inside this
    # experiment entrypoint so object_prior is produced by the dataloader.
    train_net.DatasetMapperWithImage = DatasetMapperWithImageObjectPrior

    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n%s", model)

    freeze_swin_bottom_up(model)
    os.environ["VLPART_FREEZE_SWIN_BOTTOM_UP"] = "1"

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return train_net.do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM,
        )

    install_object_prior_swanlab_logging(model)
    train_net.do_train(cfg, model, resume=args.resume)
    return train_net.do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
