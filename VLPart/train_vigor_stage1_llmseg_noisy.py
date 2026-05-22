#!/usr/bin/env python
import logging
import os

from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import build_model

import train_net
from train_vigor_stage1 import freeze_swin_bottom_up
from vlpart import add_vlpart_config
from vlpart.config_vigor_noisy import add_vigor_noisy_config
from vlpart.data.dataset_mapper_vigor_noisy import DatasetMapperWithImageVigorNoisy

import vlpart.data.datasets.vigor_noisy as vigor_noisy


logger = logging.getLogger("detectron2")


def setup(args):
    cfg = get_cfg()
    add_vlpart_config(cfg)
    add_vigor_noisy_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if "/auto" in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("/auto", f"/{file_name}")
        logger.info("OUTPUT_DIR: %s", cfg.OUTPUT_DIR)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def configure_vigor_noisy_dataset(cfg):
    vigor_noisy.set_vigor_noisy_options(
        enabled=cfg.VIGOR_NOISY.ENABLED,
        noisy_ratio=cfg.VIGOR_NOISY.NOISY_RATIO,
        nearby_topk=cfg.VIGOR_NOISY.NEARBY_TOPK,
        max_center_distance=cfg.VIGOR_NOISY.MAX_CENTER_DISTANCE,
        random_seed=cfg.VIGOR_NOISY.RANDOM_SEED,
    )


def main(args):
    train_net.DatasetMapperWithImage = DatasetMapperWithImageVigorNoisy

    cfg = setup(args)
    configure_vigor_noisy_dataset(cfg)

    model = build_model(cfg)
    logger.info("Model:\n%s", model)

    freeze_swin_bottom_up(model)
    os.environ["VLPART_FREEZE_SWIN_BOTTOM_UP"] = "1"

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=args.resume,
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
