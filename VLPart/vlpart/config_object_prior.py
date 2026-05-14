# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_object_prior_config(cfg):
    """
    Extra config used only by the VIGOR GT object-prior experiment.

    Kept in a separate file so the default VLPart config path is untouched.
    """
    cfg.MODEL.OBJECT_PRIOR = CN()
    cfg.MODEL.OBJECT_PRIOR.ENABLED = False
    cfg.MODEL.OBJECT_PRIOR.MODE = "none"
    cfg.MODEL.OBJECT_PRIOR.FPN = False
    cfg.MODEL.OBJECT_PRIOR.RPN_ROI = False
    cfg.MODEL.OBJECT_PRIOR.MASK_HEAD = False

    cfg.MODEL.OBJECT_PRIOR.FPN_ALPHA_INIT = 0.0
    cfg.MODEL.OBJECT_PRIOR.RPN_ROI_BETA_INIT = 0.1
    cfg.MODEL.OBJECT_PRIOR.MASK_ALPHA_INIT = 0.0
    cfg.MODEL.OBJECT_PRIOR.RPN_RERANK = True
