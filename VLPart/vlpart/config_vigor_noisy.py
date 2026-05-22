from detectron2.config import CfgNode as CN


def add_vigor_noisy_config(cfg):
    cfg.VIGOR_NOISY = CN()
    cfg.VIGOR_NOISY.ENABLED = False
    # Extra noisy duplicate samples relative to the full clean easy set.
    cfg.VIGOR_NOISY.NOISY_RATIO = 0.2
    # For each noisy duplicate, choose one random object among the K nearest
    # objects in the same scene and union its GT object region with the target.
    cfg.VIGOR_NOISY.NEARBY_TOPK = 1
    # Drop and redraw a noisy target when its chosen nearby object is farther
    # than this bbox-center distance in input pixels. Set <= 0 to disable.
    cfg.VIGOR_NOISY.MAX_CENTER_DISTANCE = 150.0
    cfg.VIGOR_NOISY.RANDOM_SEED = 42
