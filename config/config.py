import os

import detectron2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg

from config import paths


def get_cfg_for_vanilla(
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
    checkpoint_period: int = 1000,
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

    cfg.MODEL.WEIGHTS = (  # Model weights
        model_weights
        if model_weights is not None
        else model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch  # Images per batch
    cfg.SOLVER.BASE_LR = base_lr  # Base learning rate
    cfg.SOLVER.MAX_ITER = max_iter  # Max iteration
    cfg.SOLVER.STEPS = steps  # Steps on decaying lr
    cfg.SOLVER.WARMUP_ITERS = warmup_iters
    cfg.SOLVER.NUM_DECAYS = num_decays  # Total lr decay
    cfg.SOLVER.GAMMA = gamma  # Decay to gamma times previous lr
    cfg.SOLVER.CHECKPOINT_PERIOD = (
        checkpoint_period  # Save checkpoint every n iterations
    )

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.MASK_ON = True  # Mask on

    cfg.OUTPUT_DIR = paths.output_path

    cfg.TEST.EVAL_PERIOD = eval_period

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


def get_cfg_for_al(
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
    checkpoint_period: int = 1000,
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

    cfg.MODEL.WEIGHTS = (  # Model weights
        model_weights
        if model_weights is not None
        else model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch  # Images per batch
    cfg.SOLVER.BASE_LR = base_lr  # Base learning rate
    cfg.SOLVER.MAX_ITER = max_iter  # Max iteration
    cfg.SOLVER.STEPS = steps  # Steps on decaying lr
    cfg.SOLVER.WARMUP_ITERS = warmup_iters
    cfg.SOLVER.NUM_DECAYS = num_decays  # Total lr decay
    cfg.SOLVER.GAMMA = gamma  # Decay to gamma times previous lr
    cfg.SOLVER.CHECKPOINT_PERIOD = (
        checkpoint_period  # Save checkpoint every n iterations
    )

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.MASK_ON = True  # Mask on

    cfg.OUTPUT_DIR = paths.output_path

    cfg.TEST.EVAL_PERIOD = eval_period

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # AL-specific configs
    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"

    return cfg


def get_cfg_for_cns(
    train_dataset: str = "train",
    unlabeled_dataset: str = "test",
    test_dataset: str = "val",
    ims_per_batch: int = 2,
    ims_per_batch_labeled: int = 1,
    ims_per_batch_unlabeled: int = 1,
    base_lr: int = 0.001,
    warmup_iters: int = 200,
    model_weights: str = None,
    num_decays: int = 4,
    steps: tuple = (400, 500, 600, 700),
    gamma: float = 0.2,
    max_iter: int = 1000,
    eval_period: int = 200,
    checkpoint_period: int = 1000,
    cns_beta: float = 0.5,
    cns_w_t0: int = 1,
    cns_w_t1: int = 5000,
    cns_w_t2: int = 6000,
    cns_w_t: int = 18000,
    train_al: bool = True,
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

    cfg.MODEL.WEIGHTS = (  # Model weights
        model_weights
        if model_weights is not None
        else model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.SOLVER.BASE_LR = base_lr  # Base learning rate
    cfg.SOLVER.MAX_ITER = max_iter  # Max iteration
    cfg.SOLVER.STEPS = steps  # Steps on decaying lr
    cfg.SOLVER.WARMUP_ITERS = warmup_iters
    cfg.SOLVER.NUM_DECAYS = num_decays  # Total lr decay
    cfg.SOLVER.GAMMA = gamma  # Decay to gamma times previous lr
    cfg.SOLVER.CHECKPOINT_PERIOD = (
        checkpoint_period  # Save checkpoint every n iterations
    )

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.MASK_ON = True  # Mask on

    cfg.OUTPUT_DIR = paths.output_path

    cfg.TEST.EVAL_PERIOD = eval_period

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # CNS-specific configs
    # @martellweeks: Code taken from https://github.com/vlfom/CSD-detectron2 and modified

    ### Model parameters
    cfg.MODEL.META_ARCHITECTURE = "CNSGeneralizedRCNN"
    cfg.MODEL.ROI_HEADS.NAME = "CNSALROIHeads"

    ### Solver parameters
    # Note: with CNS enabled, the "effective" batch size (in terms of memory used) is twice larger as images get flipped
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.IMS_PER_BATCH_LABELED = ims_per_batch_labeled
    cfg.SOLVER.IMS_PER_BATCH_UNLABELED = ims_per_batch_unlabeled

    # CNS weight scheduling parameters (see their supplementary)
    # Note that here we change the notationn - T0 defines the number of iterations until the weight is zero,
    # T1 and T2 define the absolute number of iterations when to start ramp up and ramp down of the weight,
    # and T defines the target iteration when the weight is expected to finish ramping down (note: it's OK if
    # it's less than `SOLVER.NUM_ITER`)
    cfg.SOLVER.CNS_WEIGHT_SCHEDULE_RAMP_BETA = (
        cns_beta  # Base multiplier for CNS weights (not mentioned in the paper)
    )
    cfg.SOLVER.CNS_WEIGHT_SCHEDULE_RAMP_T0 = (
        cns_w_t0  # Train for one iteration without CNS loss
    )
    cfg.SOLVER.CNS_WEIGHT_SCHEDULE_RAMP_T1 = cns_w_t1
    cfg.SOLVER.CNS_WEIGHT_SCHEDULE_RAMP_T2 = cns_w_t2
    # Note: even though `T` represents the total number of iterations, it's safe to continue training after `T` iters
    cfg.SOLVER.CNS_WEIGHT_SCHEDULE_RAMP_T = cns_w_t

    cfg.SOLVER.TRAIN_AL_SCORING_MODULES = train_al

    cfg.DATASETS.TRAIN = (
        train_dataset,
    )  # Note: only a single dataset is currently supported
    # Note: only a single dataset is currently supported; also note: this is not used when `cfg.DATASETS.MODE`
    # is "RANDOM_SPLIT"
    cfg.DATASETS.TRAIN_UNLABELED = (unlabeled_dataset,)

    # Only VOC and COCO are currently supported for evaluation; also only a **single** evaluation dataset
    # is supported (for visualization reasons; if you turn it off, multiple datasets should work)
    cfg.DATASETS.TEST = (test_dataset,)

    # Defines if two separate datasets should be used as labeled and unlabeled data, or a single dataset must
    # be split into labeled and unlabeled parts; supported values: "CROSS_DATASET", "RANDOM_SPLIT"
    cfg.DATASETS.MODE = "CROSS_DATASET"

    # Required if `cfg.DATASETS.MODE` is "RANDOM_SPLIT".
    # Defines whether to load the split from the file with the path provided, or to generate a new split:
    # - if True, loads the split from `cfg.DATASETS.RANDOM_SPLIT_PATH`, see its comments below;
    # - if False, uses `cfg.DATASETS.SUP_PERCENT` and `cfg.DATASETS.RANDOM_SPLIT_SEED` to generate
    # a new split using `cfg.DATASETS.TRAIN` dataset
    cfg.DATASETS.SPLIT_USE_PREDEFINED = False

    # Required if `cfg.DATASETS.MODE` is "RANDOM_SPLIT".
    # Defines path to the file that either (1) contains a pre-defined list of image indices to use as labeled data
    # or (2) should be used to output the generated split.
    # The file must contain a stringified Python list of strings of the corresponding dataset's images `image_id`s
    # e.g.: ['000073', '000194', '000221']; see datasets/voc_splits/example_split.txt.
    # `image_id` is an invariant across many D2-formatted datasets. See for example:
    # `_cityscapes_files_to_dict()`, `load_voc_instances()`, `load_coco_json()`.
    # TODO: add example
    cfg.DATASETS.SPLIT_PATH = None

    # (optional) % of the images from the dataset to use as supervised data;
    # must be set if `cfg.DATASETS.SPLIT_USE_PREDEFINED` is True

    cfg.DATASETS.SPLIT_SUP_PERCENT = None
    # (optional) random seed to use for `np.random.seed` when generating the data split, it is necessary
    # for reproducibility and to make sure that each GPU uses the same data split;
    # must be set if `cfg.DATASETS.SPLIT_USE_PREDEFINED` is True
    cfg.DATASETS.SPLIT_SEED = None

    cfg.VIS_TEST = False

    return cfg
