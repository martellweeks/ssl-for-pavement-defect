import os

import detectron2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg

from config import paths

cfg = get_cfg()

cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("val",)
cfg.DATALOADER.NUM_WORKERS = 2  # Dataloader workers
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)  # Model weights
cfg.SOLVER.IMS_PER_BATCH = 5  # Images per batch
cfg.SOLVER.BASE_LR = 0.001  # Base learning rate
cfg.SOLVER.MAX_ITER = 10000  # Max iteration
cfg.SOLVER.STEPS = (
    2000,
    3000,
    4000,
    5000,
)  # Steps on decaying lr
cfg.SOLVER.NUM_DECAYS = 4  # Total lr decay
cfg.SOLVER.GAMMA = 0.2  # Decay to gamma times previous lr
cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoint every 1000 iterations
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.MASK_ON = True  # Mask
cfg.OUTPUT_DIR = paths.output_path
cfg.TEST.EVAL_PERIOD = 2000
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
