#!/usr/bin/env python
import logging
import os

import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch
from detectron2.modeling import build_model

from train_net import do_test, do_train, setup


logger = logging.getLogger("detectron2")


def _count_params(module):
    total = 0
    trainable = 0
    for param in module.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return total, trainable


def freeze_swin_bottom_up(model):
    """Freeze only the Swin bottom_up backbone, keeping FPN/RPN/ROI/mask heads trainable."""
    backbone = getattr(model, "backbone", None)
    bottom_up = getattr(backbone, "bottom_up", None)
    if bottom_up is None:
        raise AttributeError("Expected model.backbone.bottom_up for Swin+FPN VLPart model")

    for param in bottom_up.parameters():
        param.requires_grad = False

    # Keep stochastic backbone layers deterministic after train_net.do_train() calls model.train().
    original_train = bottom_up.train

    def frozen_train(mode=True):
        original_train(False)
        return bottom_up

    bottom_up.train = frozen_train
    bottom_up.eval()

    total, trainable = _count_params(model)
    frozen_total, frozen_trainable = _count_params(bottom_up)
    logger.info(
        "Frozen Swin bottom_up parameters: trainable=%d / total=%d",
        frozen_trainable,
        frozen_total,
    )
    logger.info(
        "Whole model trainable parameters after freeze: trainable=%d / total=%d",
        trainable,
        total,
    )


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n%s", model)

    freeze_swin_bottom_up(model)
    os.environ["VLPART_FREEZE_SWIN_BOTTOM_UP"] = "1"

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM,
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


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
