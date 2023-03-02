import logging
import os
from datetime import datetime

import detectron2
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

from config import paths
from config.config import cfg as cfg
from src.model import hooks
from src.model.al_scoring_head import ALScoringROIHeads


def train():
    logger = setup_logger()
    fileHandler = logging.FileHandler(f"log_{datetime.now()}")
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)

    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    logger.info("Check versions...")
    logger.info(f"torch:  {TORCH_VERSION} ; cuda:  {CUDA_VERSION}")
    logger.info(f"detectron2: {detectron2.__version__}")

    register_coco_instances("train", {}, paths.train_anns_path, paths.train_data_path)
    register_coco_instances("val", {}, paths.val_anns_path, paths.val_data_path)
    register_coco_instances("test", {}, paths.test_anns_path, paths.test_data_path)

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"

    train_ds = DatasetCatalog.get("train")
    val_ds = DatasetCatalog.get("val")
    metadata = MetadataCatalog.get("train")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = hooks.MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    logger.info("Start training...")
    trainer.train()

    logger.info("Training completed")
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save(paths.output_model_filename)
    logger.info("Final model saved")
