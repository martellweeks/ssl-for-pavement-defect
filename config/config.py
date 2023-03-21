import os

import detectron2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg

from config import paths


def get_default_cfg():
    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.DATALOADER.NUM_WORKERS = 2  # Dataloader workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )  # Model weights
    cfg.SOLVER.IMS_PER_BATCH = 5  # Images per batch
    cfg.SOLVER.BASE_LR = 0.00005  # Base learning rate
    cfg.SOLVER.MAX_ITER = 320  # Max iteration
    cfg.SOLVER.STEPS = (
        500,
        600,
        700,
        800,
    )  # Steps on decaying lr
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.NUM_DECAYS = 0  # Total lr decay
    cfg.SOLVER.GAMMA = 0.2  # Decay to gamma times previous lr
    cfg.SOLVER.CHECKPOINT_PERIOD = 100  # Save checkpoint every 1000 iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.MASK_ON = True  # Mask
    cfg.OUTPUT_DIR = paths.output_path
    cfg.TEST.EVAL_PERIOD = 320
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def get_cfg_with_exp_setup(
    train_dataset: str = "train",
    ims_per_batch: int = 5,
    base_lr: int = 0.001,
    warmup_iters: int = 200,
    model_weights: str = None,
    num_decays: int = 4,
    steps: tuple = (400, 500, 600, 700),
    gamma: float = 0.2,
    max_iter: int = 1000,
    eval_period: int = 200,
):
    assert len(steps) == num_decays

    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = ("val",)
    cfg.DATALOADER.NUM_WORKERS = 2  # Dataloader workers
    cfg.MODEL.WEIGHTS = (
        model_weights
        if model_weights is not None
        else model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )  # Model weights
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch  # Images per batch
    cfg.SOLVER.BASE_LR = base_lr  # Base learning rate
    cfg.SOLVER.MAX_ITER = max_iter  # Max iteration
    cfg.SOLVER.STEPS = steps  # Steps on decaying lr
    cfg.SOLVER.WARMUP_ITERS = warmup_iters
    cfg.SOLVER.NUM_DECAYS = num_decays  # Total lr decay
    cfg.SOLVER.GAMMA = gamma  # Decay to gamma times previous lr
    cfg.SOLVER.CHECKPOINT_PERIOD = 100  # Save checkpoint every 100 iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.MASK_ON = True  # Mask
    cfg.OUTPUT_DIR = paths.output_path
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg
