from detectron2.config import CfgNode as CN


def add_vigor_noisy_config(cfg):
    cfg.VIGOR_NOISY = CN()
    cfg.VIGOR_NOISY.ENABLED = False
    cfg.VIGOR_NOISY.NOISY_RATIO = 0.5
    cfg.VIGOR_NOISY.TOPK_MASKS_DIR = "/opt/data/private/LLMSeg/vis_output_object_topk_trainset/topk_masks"
    cfg.VIGOR_NOISY.TOPK_RANK = 1
    cfg.VIGOR_NOISY.DIFFICULTY = "easy"
    cfg.VIGOR_NOISY.INSTRUCTION_INDEX = 0
    cfg.VIGOR_NOISY.OBJECT_IOU_THRESHOLD = 0.5
    cfg.VIGOR_NOISY.AFF_COVERAGE_THRESHOLD = 0.7
    cfg.VIGOR_NOISY.RANDOM_SEED = 42
